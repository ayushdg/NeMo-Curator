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

import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.file_utils import get_fs


@dataclass
class InterleavedSampleDuplicatesRemovalStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Remove duplicate interleaved samples based on pre-computed sample IDs."""

    ids_to_remove_path: str
    id_field: str = CURATOR_DEDUP_ID_STR
    duplicate_id_field: str = CURATOR_DEDUP_ID_STR
    sample_id_field: str = "sample_id"
    read_kwargs: dict[str, Any] | None = None
    drop_id_field: bool = True

    def __post_init__(self) -> None:
        super().__init__()
        self.name = "InterleavedSampleDuplicatesRemovalStage"
        self.read_kwargs = self.read_kwargs.copy() if self.read_kwargs else {}
        self._removal_fs = get_fs(self.ids_to_remove_path, self.read_kwargs.get("storage_options", {}))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = [self.sample_id_field]
        if not self.drop_id_field:
            output_fields.append(self.id_field)
        return ["data"], output_fields

    def _has_removal_files(self) -> bool:
        if not self._removal_fs.exists(self.ids_to_remove_path):
            return False
        if self._removal_fs.isdir(self.ids_to_remove_path):
            return any(path.endswith(".parquet") for path in self._removal_fs.find(self.ids_to_remove_path))
        return True

    def _read_removal_subset(self, min_id: int, max_id: int) -> pd.DataFrame:
        if not self._has_removal_files():
            return pd.DataFrame({self.duplicate_id_field: pd.Series(dtype="int64")})
        return pd.read_parquet(
            self.ids_to_remove_path,
            filters=[(self.duplicate_id_field, ">=", min_id), (self.duplicate_id_field, "<=", max_id)],
            columns=[self.duplicate_id_field],
            **self.read_kwargs,
        )

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas()
        if len(df) == 0:
            return task
        if self.id_field not in df.columns:
            msg = f"Input interleaved batch is missing required id field '{self.id_field}'"
            raise ValueError(msg)
        if self.sample_id_field not in df.columns:
            msg = f"Input interleaved batch is missing required sample field '{self.sample_id_field}'"
            raise ValueError(msg)

        min_max_t0 = time.perf_counter()
        min_id = int(df[self.id_field].min())
        max_id = int(df[self.id_field].max())
        min_max_time = time.perf_counter() - min_max_t0

        read_dupes_t0 = time.perf_counter()
        removal_df = self._read_removal_subset(min_id, max_id)
        read_dupes_time = time.perf_counter() - read_dupes_t0

        remove_t0 = time.perf_counter()
        removal_ids = set(removal_df[self.duplicate_id_field].tolist())
        if removal_ids:
            duplicate_rows = df[self.id_field].isin(removal_ids)
            sample_ids_to_drop = set(df.loc[duplicate_rows, self.sample_id_field].tolist())
            df_kept = df[~df[self.sample_id_field].isin(sample_ids_to_drop)]
        else:
            sample_ids_to_drop = set()
            df_kept = df
        removal_time = time.perf_counter() - remove_t0

        if self.drop_id_field and self.id_field in df_kept.columns:
            df_kept = df_kept.drop(columns=[self.id_field])

        self._log_metrics(
            {
                "input_df_min_max_time": min_max_time,
                "read_dupes_time": read_dupes_time,
                "id_removal_time": removal_time,
            }
        )

        return InterleavedBatch(
            task_id=f"removal_{task.task_id}",
            dataset_name=task.dataset_name,
            data=df_kept.reset_index(drop=True),
            _metadata={
                **task._metadata,
                "num_removed": len(sample_ids_to_drop),
                "num_samples_removed": len(sample_ids_to_drop),
                "num_rows_in": len(df),
                "num_rows_out": len(df_kept),
            },
            _stage_perf=task._stage_perf,
        )
