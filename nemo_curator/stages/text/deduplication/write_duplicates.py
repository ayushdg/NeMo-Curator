# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write out the duplicate documents identified by fuzzy deduplication.

This is the mirror image of the removal stage: instead of dropping the duplicate ids, it writes the
full duplicate documents out, grouped by ``_duplicate_group_id``. It is driven by the shuffled
``(id, group)`` partitions produced by :class:`ShuffleDuplicateGroupsStage` (one partition per task)
and uses a driver-built :class:`ReverseIdIndex` to look up which original input file groups hold the
documents for a given set of ids, re-reads those groups, and GPU-merges them with the duplicate
table to attach the group id.
"""

from typing import Any, Literal

import cudf
import numpy as np
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.fuzzy.reverse_id_index import ReverseIdIndex
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs

__all__ = ["ReverseIdIndex", "WriteTextDuplicatesStage"]


class WriteTextDuplicatesStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """Write the duplicate documents for one shuffled ``(id, group)`` partition.

    For each input task (one ``FuzzyDuplicateGroups`` partition file):
      1. cudf-read the partition's ``(id, group)`` table.
      2. Reverse-look-up the original input file groups holding those ids.
      3. Re-read each group, re-assign the contiguous id range positionally, and GPU-merge with the
         duplicate table to keep only duplicate documents and attach ``_duplicate_group_id``.
      4. Sort by group (then id) and write a single output file whose rows are complete groups.

    The stage reproduces ids positionally from the driver-built index, so it does **not** need the
    id generator actor.
    """

    def __init__(  # noqa: PLR0913
        self,
        reverse_index: ReverseIdIndex,
        output_path: str,
        input_filetype: Literal["parquet", "jsonl"] = "parquet",
        id_field: str = CURATOR_DEDUP_ID_STR,
        duplicate_group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD,
        fields: list[str] | None = None,
        output_fields: list[str] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        duplicate_read_kwargs: dict[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        reverse_index
            Driver-built mapping of id range -> input file group. See :class:`ReverseIdIndex`.
        output_path
            Directory to write the duplicate documents to. Files are written under
            ``<output_path>/<stage name>``, one per input partition.
        input_filetype
            Format of the original input dataset ("parquet" or "jsonl").
        id_field
            Column name holding the ``_curator_dedup_id`` in the duplicate table.
        duplicate_group_field
            Column name holding the duplicate group id in the duplicate table.
        fields
            Input columns to read (content). If None, all columns are read.
        output_fields
            Columns to write out. If None, all read columns plus the group id are written.
        read_kwargs
            Extra kwargs for reading the original input files (cudf).
        write_kwargs
            Extra kwargs for writing the output files.
        duplicate_read_kwargs
            Extra kwargs for reading the duplicate ``(id, group)`` partition files (e.g.
            storage_options).
        """
        self.name = self.__class__.__name__
        self.resources = Resources(gpus=1.0)

        self.reverse_index = reverse_index
        self.input_filetype = input_filetype
        self.id_field = id_field
        self.duplicate_group_field = duplicate_group_field
        self.fields = fields
        self.output_fields = output_fields
        self.read_kwargs = read_kwargs.copy() if read_kwargs else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs else {}
        self.duplicate_read_kwargs = duplicate_read_kwargs.copy() if duplicate_read_kwargs else {}

        if self.input_filetype not in ("parquet", "jsonl"):
            msg = f"Invalid input_filetype: {self.input_filetype}"
            raise ValueError(msg)

        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options", {}))
        self.output_path = self.output_fs.sep.join([output_path, self.name])
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))

        # We reproduce ids positionally from the reverse index, so no id generator is needed; the
        # DeduplicationIO read primitives only touch the id generator when assign_id=True.
        self.id_generator = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _read_duplicate_partition(self, files: list[str]) -> cudf.DataFrame:
        return cudf.read_parquet(
            files,
            columns=[self.id_field, self.duplicate_group_field],
            **self.duplicate_read_kwargs,
        )

    def _read_input_group(self, files: list[str]) -> cudf.DataFrame:
        if self.input_filetype == "parquet":
            read_kwargs = dict(self.read_kwargs)
            if self.fields is not None:
                read_kwargs["columns"] = self.fields
            return self.read_parquet(filepath=files, assign_id=False, **read_kwargs)
        return self.read_jsonl(filepath=files, columns=self.fields, assign_id=False, **self.read_kwargs)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        # 1) Read this shuffled duplicate partition: (id, group). Read in full (no range filter).
        dup_df = self._read_duplicate_partition(task.data)
        if len(dup_df) == 0:
            return self._empty_output(task)

        dup_ids = dup_df[self.id_field].to_numpy()

        # 2) Reverse lookup: which original input file groups hold these ids.
        group_indices = self.reverse_index.group_indices_for_ids(dup_ids)

        # 3) Read each group, reproduce ids positionally, GPU-merge to attach the group id.
        matched = []
        num_input_files = 0
        for idx in group_indices:
            files = self.reverse_index.files(idx)
            min_id = self.reverse_index.min_id(idx)
            expected_n = self.reverse_index.max_id(idx) - min_id + 1
            num_input_files += len(files)

            gdf = self._read_input_group(files)
            if len(gdf) != expected_n:
                msg = (
                    f"{self.name}: input group {files} produced {len(gdf)} rows but the id generator "
                    f"registered {expected_n} ids for it during minhash (range [{min_id}, "
                    f"{self.reverse_index.max_id(idx)}]). The read diverged from the minhash read "
                    f"(different files, order, or columns)."
                )
                raise RuntimeError(msg)

            gdf[self.id_field] = np.arange(min_id, min_id + len(gdf))
            gdf = gdf.merge(dup_df, on=self.id_field, how="inner")
            if len(gdf) > 0:
                matched.append(gdf)

        if not matched:
            return self._empty_output(task)

        merged = cudf.concat(matched, ignore_index=True) if len(matched) > 1 else matched[0]
        merged = merged.sort_values(by=[self.duplicate_group_field, self.id_field], ignore_index=True)
        if self.output_fields is not None:
            merged = merged[self.output_fields]

        output_file = self.output_fs.sep.join([self.output_path, self._output_name(task)])
        self.write_parquet(df=merged, filepath=output_file, **self.write_kwargs)
        logger.debug(
            "{} partition={} groups={} duplicates_written={} input_files_read={}",
            self.name,
            task.task_id,
            len(group_indices),
            len(merged),
            num_input_files,
        )

        return FileGroupTask(
            task_id=f"duplicates_{task.task_id}",
            dataset_name=f"{task.dataset_name}_duplicates",
            data=[output_file],
            _metadata={
                **task._metadata,
                "num_duplicates_written": len(merged),
                "num_input_groups_read": len(group_indices),
                "num_input_files_read": num_input_files,
                "storage_options": self.write_kwargs.get("storage_options"),
            },
            _stage_perf=task._stage_perf,
        )

    def _output_name(self, task: FileGroupTask) -> str:
        if len(task.data) == 1:
            name = task.data[0].split("/")[-1]
            return name if name.endswith(".parquet") else f"{name}.parquet"
        return f"{task._uuid}.parquet"

    def _empty_output(self, task: FileGroupTask) -> FileGroupTask:
        return FileGroupTask(
            task_id=f"duplicates_{task.task_id}",
            dataset_name=f"{task.dataset_name}_duplicates",
            data=[],
            _metadata={
                **task._metadata,
                "num_duplicates_written": 0,
                "num_input_groups_read": 0,
                "num_input_files_read": 0,
            },
            _stage_perf=task._stage_perf,
        )
