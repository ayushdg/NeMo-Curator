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

"""Write duplicate documents grouped by ``_duplicate_group_id`` using a disk-based, map-only
(external-sort / range-merge) shuffle instead of a collective shuffle or a reverse-lookup that
re-reads the whole input per output partition.

Three map stages (all embarrassingly parallel, no reduce primitive):

1. ``SortDuplicatesByIdStage`` — sort the connected-components ``(id, group)`` output by
   ``_curator_dedup_id`` so per-input-group range reads get parquet predicate pushdown.
2. ``MaterializeDuplicateRunsStage`` — input-driven: read each original input file group ONCE,
   attach ``_duplicate_group_id`` (range-read the sorted CC), keep duplicate rows, sort by group,
   write ONE run file per map task (group-sorted, row-grouped).
3. ``MergeDuplicateGroupsStage`` — map over B ``group_id`` ranges: range-read ``[lo, hi)`` from all
   run files via pushdown (sorted runs ⇒ skips non-overlapping row groups), concat, write one
   group-complete output file per range.

The cross-cutting glue (run-file list, ``group_id`` cut-points, the B range tasks) lives on the
driver in the workflow, so no stage needs a Curator reduce/shuffle primitive.
"""

from typing import Any, Literal

import cudf
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import ray
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs

SORTED_CC_SUBDIR = "ConnectedComponentsSortedById"
DUPLICATE_RUNS_SUBDIR = "DuplicateRuns"
DUPLICATE_GROUPS_SUBDIR = "DuplicateGroups"

DEFAULT_ROW_GROUP_SIZE = 1_000_000
DEFAULT_MERGE_READ_CHUNK = 512  # run files read per cudf.read_parquet call in the merge
MIN_RUN_ROW_GROUP_ROWS = 256  # floor for the adaptive per-run row-group size


def _to_large_string(table: "pa.Table") -> "pa.Table":
    """Cast string/binary columns to large_string/large_binary (64-bit offsets).

    pyarrow's default ``string`` type uses 32-bit offsets, so a single string column chunk caps at
    ~2 GB of characters; concatenating a giant duplicate bucket's ``raw_content`` overflows it
    ("offset overflow while concatenating arrays"). large_string lifts that to 64-bit, so the only
    limit is host memory.
    """
    fields = []
    changed = False
    for field in table.schema:
        if pa.types.is_string(field.type):
            fields.append(pa.field(field.name, pa.large_string()))
            changed = True
        elif pa.types.is_binary(field.type):
            fields.append(pa.field(field.name, pa.large_binary()))
            changed = True
        else:
            fields.append(field)
    return table.cast(pa.schema(fields)) if changed else table


class SortDuplicatesByIdStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """Sort each connected-components ``(id, group)`` file by ``_curator_dedup_id``.

    Output files are written with bounded row groups so each row group's id min/max is tight,
    enabling predicate pushdown when the materialize stage range-reads by id.
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        id_field: str = CURATOR_DEDUP_ID_STR,
        group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
    ):
        self.name = self.__class__.__name__
        self.resources = Resources(gpus=1.0)
        self.id_field = id_field
        self.group_field = group_field
        self.read_kwargs = read_kwargs.copy() if read_kwargs else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs else {}
        self.row_group_size = row_group_size
        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options", {}))
        self.output_path = self.output_fs.sep.join([output_path, SORTED_CC_SUBDIR])
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))
        self.id_generator = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.id_field, self.group_field]

    def process(self, task: FileGroupTask) -> FileGroupTask:
        df = self.read_parquet(
            filepath=task.data, assign_id=False, columns=[self.id_field, self.group_field], **self.read_kwargs
        )
        df = df.sort_values(self.id_field, ignore_index=True)
        output_file = self.output_fs.sep.join([self.output_path, f"{task._uuid}.parquet"])
        write_kwargs = {**self.write_kwargs, "row_group_size_rows": self.row_group_size, "index": False}
        self.write_parquet(df=df, filepath=output_file, **write_kwargs)
        return FileGroupTask(
            task_id=f"sortcc_{task.task_id}",
            dataset_name=f"{task.dataset_name}_sorted",
            data=[output_file],
            _metadata={**task._metadata, "num_rows": len(df)},
            _stage_perf=task._stage_perf,
        )


class MaterializeDuplicateRunsStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """Read original input file groups once and write group-sorted runs of their duplicate rows.

    Input is one ``FileGroupTask`` per original input file group (``task.data`` = that group's files,
    which are the files actually read). Each group's id range is looked up from the IdGenerator actor
    (same pattern as ``InterleavedDuplicatesRemovalStage``), the contiguous ``_curator_dedup_id`` is
    reproduced positionally, the sorted CC is range-read to attach ``group_id``, and duplicate rows
    are kept via inner join.

    Use ``batch_size`` to coarsen the number of run files: the executor feeds ``batch_size`` input
    groups to each ``process_batch`` call, which merges them and writes ONE group-sorted run — so
    ``n_map = ceil(num_input_groups / batch_size)``.
    """

    def __init__(  # noqa: PLR0913
        self,
        sorted_cc_path: str,
        output_path: str,
        input_filetype: Literal["parquet", "jsonl"] = "parquet",
        id_field: str = CURATOR_DEDUP_ID_STR,
        group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD,
        fields: list[str] | None = None,
        output_fields: list[str] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        cc_read_kwargs: dict[str, Any] | None = None,
        num_output_partitions: int = 256,
    ):
        self.name = self.__class__.__name__
        self.resources = Resources(gpus=1.0)
        self.sorted_cc_path = sorted_cc_path
        self.input_filetype = input_filetype
        self.id_field = id_field
        self.group_field = group_field
        self.fields = fields
        self.output_fields = output_fields
        self.read_kwargs = read_kwargs.copy() if read_kwargs else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs else {}
        self.cc_read_kwargs = cc_read_kwargs.copy() if cc_read_kwargs else {}
        # Size run row groups to ~1 / num_output_partitions of the group_id space so the merge's
        # predicate pushdown can prune (a run spans the whole group_id range, so few/large row
        # groups would force the merge to read every run for every output range).
        self.num_output_partitions = num_output_partitions
        if self.input_filetype not in ("parquet", "jsonl"):
            msg = f"Invalid input_filetype: {self.input_filetype}"
            raise ValueError(msg)
        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options", {}))
        self.output_path = self.output_fs.sep.join([output_path, DUPLICATE_RUNS_SUBDIR])
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))
        self.id_generator = None

    def setup(self, _worker_metadata=None) -> None:  # noqa: ANN001
        from nemo_curator.stages.deduplication.id_generator import get_id_generator_actor

        try:
            self.id_generator = get_id_generator_actor()
        except ValueError as e:
            err_msg = (
                f"{self.name} requires the IdGenerator actor used during minhash to be loaded "
                "(e.g. create_id_generator_actor(filepath=<fuzzy_id_generator.json>)) so it can look "
                "up each input file group's _curator_dedup_id range."
            )
            raise ValueError(err_msg) from e

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _read_input_group(self, files: list[str]) -> cudf.DataFrame:
        if self.input_filetype == "parquet":
            read_kwargs = dict(self.read_kwargs)
            if self.fields is not None:
                read_kwargs["columns"] = self.fields
            return self.read_parquet(filepath=files, assign_id=False, **read_kwargs)
        return self.read_jsonl(filepath=files, columns=self.fields, assign_id=False, **self.read_kwargs)

    def _read_cc_subset(self, min_id: int, max_id: int) -> cudf.DataFrame:
        return cudf.read_parquet(
            self.sorted_cc_path,
            filters=[(self.id_field, ">=", min_id), (self.id_field, "<=", max_id)],
            columns=[self.id_field, self.group_field],
            **self.cc_read_kwargs,
        )

    def _materialize_group(self, files: list[str]) -> cudf.DataFrame | None:
        """Read one input file group, reproduce its ids, and return its duplicate rows (or None)."""
        min_id, max_id = ray.get(self.id_generator.get_batch_range.remote(files=files, key=None))
        df = self._read_input_group(files)
        expected_n = max_id - min_id + 1
        if len(df) != expected_n:
            msg = (
                f"{self.name}: input group {files} produced {len(df)} rows but the id generator "
                f"registered {expected_n} ids (range [{min_id}, {max_id}]). Read diverged from minhash "
                "(different files, order, columns, or blocksize)."
            )
            raise RuntimeError(msg)
        df[self.id_field] = np.arange(min_id, min_id + len(df))
        cc = self._read_cc_subset(min_id, max_id)
        if len(cc) == 0:
            return None
        # Drop non-duplicate rows (and their potentially large content columns) BEFORE the join so
        # the merge operates only on the ~duplicate subset, keeping peak GPU memory bounded.
        df = df[df[self.id_field].isin(cc[self.id_field])]
        if len(df) == 0:
            return None
        merged = df.merge(cc, on=self.id_field, how="inner")
        return merged if len(merged) > 0 else None

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        """Materialize a batch of input file groups into ONE group-sorted run file."""
        if self.id_generator is None:
            msg = f"{self.name}: IdGenerator actor not initialized. Call setup() first."
            raise RuntimeError(msg)

        parts = []
        for task in tasks:
            merged = self._materialize_group(task.data)
            if merged is not None:
                parts.append(merged)
        if not parts:
            return []

        run = cudf.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        run = run.sort_values([self.group_field, self.id_field], ignore_index=True)
        if self.output_fields is not None:
            run = run[self.output_fields]

        # Many small, group_id-sorted row groups so the merge can prune by group_id range.
        run_row_group = max(MIN_RUN_ROW_GROUP_ROWS, len(run) // self.num_output_partitions)
        output_file = self.output_fs.sep.join([self.output_path, f"run.{tasks[0]._uuid}.parquet"])
        write_kwargs = {**self.write_kwargs, "row_group_size_rows": run_row_group, "index": False}
        self.write_parquet(df=run, filepath=output_file, **write_kwargs)
        logger.debug("{} batch_groups={} dup_rows={} row_group={}", self.name, len(tasks), len(run), run_row_group)
        return [
            FileGroupTask(
                task_id=f"run_{tasks[0].task_id}",
                dataset_name=f"{tasks[0].dataset_name}_runs",
                data=[output_file],
                _metadata={
                    "num_input_groups": len(tasks),
                    "num_duplicate_rows": len(run),
                    "storage_options": self.write_kwargs.get("storage_options"),
                },
                _stage_perf=tasks[0]._stage_perf,
            )
        ]

    def process(self, task: FileGroupTask) -> list[FileGroupTask]:
        # The executor drives this stage via process_batch; this single-task entrypoint delegates
        # so direct calls behave identically (one input group -> one run, modulo empties).
        return self.process_batch([task])


class MergeDuplicateGroupsStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Merge one ``group_id`` range across all run files into a single group-complete output file.

    **CPU / pyarrow** stage: range-reads ``group_id ∈ [group_lo, group_hi)`` from the runs (parquet
    predicate pushdown on row-group stats), accumulates in host memory, sorts by ``group_id``, and
    writes one file. Running on the host (TBs of RAM, no per-column/size limit) is what lets a single
    *giant* duplicate bucket — which is atomic and cannot be split across output files — write fine
    where a GPU cuDF write would OOM. Every member of a given ``group_id`` falls in exactly one
    range, so all of a group's members land together in one output file. Bounds come from
    ``task._metadata`` (``group_lo``/``group_hi``, either may be None for the open end;
    ``partition_index``).
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        id_field: str = CURATOR_DEDUP_ID_STR,
        group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD,
        storage_options: dict[str, Any] | None = None,
        row_group_size: int = DEFAULT_ROW_GROUP_SIZE,
        read_chunk_size: int = DEFAULT_MERGE_READ_CHUNK,
        cpus: float = 4.0,
    ):
        self.name = self.__class__.__name__
        self.resources = Resources(cpus=cpus)
        self.id_field = id_field
        self.group_field = group_field
        self.storage_options = storage_options or {}
        self.row_group_size = row_group_size
        self.read_chunk_size = max(1, read_chunk_size)
        self.output_fs = get_fs(output_path, self.storage_options)
        self.output_path = self.output_fs.sep.join([output_path, DUPLICATE_GROUPS_SUBDIR])
        create_or_overwrite_dir(self.output_path, storage_options=self.storage_options)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def _pa_filesystem(self):  # noqa: ANN202
        # Native local filesystem when no storage_options; otherwise wrap the fsspec fs for pyarrow.
        if not self.storage_options:
            return None
        import pyarrow.fs as pa_fs

        return pa_fs.PyFileSystem(pa_fs.FSSpecHandler(self.output_fs))

    def _range_filter(self, group_lo: int | None, group_hi: int | None):  # noqa: ANN202
        field = pa_ds.field(self.group_field)
        expr = None
        if group_lo is not None:
            expr = field >= int(group_lo)
        if group_hi is not None:
            upper = field < int(group_hi)
            expr = upper if expr is None else (expr & upper)
        return expr

    def process(self, task: FileGroupTask) -> FileGroupTask:
        if not task.data:
            return self._empty_output(task)
        group_lo = task._metadata.get("group_lo")
        group_hi = task._metadata.get("group_hi")
        partition_index = task._metadata.get("partition_index", task.task_id)
        filt = self._range_filter(group_lo, group_hi)
        fs = self._pa_filesystem()

        # Range-read the runs in chunks (pushdown prunes the row groups outside [lo, hi)) and
        # accumulate in host memory. Host RAM has no per-column/size limit, so a giant bucket's
        # partition writes fine. A 0-row range produces no file (an empty parquet would break a
        # later multi-file read of the output dir).
        tables = []
        for i in range(0, len(task.data), self.read_chunk_size):
            chunk = task.data[i : i + self.read_chunk_size]
            tbl = pa_ds.dataset(chunk, format="parquet", filesystem=fs).to_table(filter=filt)
            if tbl.num_rows > 0:
                # Cast to large_string BEFORE concat/sort: sort_by's `take` combines chunks into one
                # contiguous array, which overflows 32-bit `string` offsets for a >2 GB bucket.
                tables.append(_to_large_string(tbl))
        if not tables:
            return self._empty_output(task)
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        table = table.sort_by([(self.group_field, "ascending"), (self.id_field, "ascending")])

        output_file = self.output_fs.sep.join([self.output_path, f"part.{partition_index}.parquet"])
        pq.write_table(table, output_file, filesystem=fs, row_group_size=self.row_group_size)
        logger.debug(
            "{} partition={} group_range=[{},{}) rows={}",
            self.name, partition_index, group_lo, group_hi, table.num_rows,
        )
        return FileGroupTask(
            task_id=f"dupgroups_{partition_index}",
            dataset_name=f"{task.dataset_name}_grouped",
            data=[output_file],
            _metadata={
                "partition_index": partition_index,
                "group_lo": group_lo,
                "group_hi": group_hi,
                "num_rows": table.num_rows,
                "storage_options": self.storage_options or None,
            },
            _stage_perf=task._stage_perf,
        )

    def _empty_output(self, task: FileGroupTask) -> FileGroupTask:
        return FileGroupTask(
            task_id=f"dupgroups_{task._metadata.get('partition_index', task.task_id)}",
            dataset_name=f"{task.dataset_name}_grouped",
            data=[],
            _metadata={**task._metadata, "num_rows": 0},
            _stage_perf=task._stage_perf,
        )
