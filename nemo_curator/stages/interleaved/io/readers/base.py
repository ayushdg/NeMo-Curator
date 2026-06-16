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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import ray
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.utils.deduplication import sample_ordering
from nemo_curator.stages.interleaved.utils.schema import align_table, reconcile_schema, resolve_schema
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class BaseInterleavedReader(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Base contract for interleaved readers.

    By default (``schema=None``) user-added passthrough columns are preserved
    and only reserved-column types are reconciled via ``reconcile_schema``.

    If *schema* is set explicitly, every output table is strictly aligned to it
    (missing columns become typed nulls, extra columns are dropped).

    Use *schema_overrides* to add or override individual field types relative to
    ``INTERLEAVED_SCHEMA`` while keeping strict alignment:

    .. code-block:: python

        reader = InterleavedParquetReader(
            "data.parquet",
            schema_overrides={"url": pa.string(), "timestamp": pa.int64()},
        )
    """

    read_kwargs: dict[str, Any] = field(default_factory=dict)
    schema: pa.Schema | None = None
    schema_overrides: dict[str, pa.DataType] | None = None
    _generate_ids: bool = False
    _assign_ids: bool = False
    name: str = "base_interleaved_reader"

    def __post_init__(self) -> None:
        if self._generate_ids and self._assign_ids:
            msg = "Cannot generate and assign IDs at the same time"
            raise ValueError(msg)
        if self.schema is not None or self.schema_overrides is not None:
            self.schema = resolve_schema(self.schema, self.schema_overrides)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = ["sample_id", "position", "modality"]
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

            output_fields.append(CURATOR_DEDUP_ID_STR)
        return ["data"], output_fields

    def setup(self, _: WorkerMetadata | None = None) -> None:
        if self._generate_ids or self._assign_ids:
            from nemo_curator.stages.deduplication.id_generator import get_id_generator_actor

            try:
                self.id_generator = get_id_generator_actor()
            except ValueError:
                msg = (
                    "ID generator is required when self._generate_ids or self._assign_ids is True, "
                    "and the actor 'id_generator' does not exist. Please start the id_generator actor."
                )
                raise RuntimeError(msg) from None

    def _align_output(self, table: pa.Table) -> pa.Table:
        """Reconcile or align *table* to the declared schema."""
        if self.schema is not None:
            return align_table(table, self.schema)
        return table.cast(reconcile_schema(table.schema))

    def _sample_ids_for_ids(self, table: pa.Table) -> list[str]:
        return sample_ordering(table, "sample_id")

    def _append_sample_ids(self, table: pa.Table, sample_ids: list[str], min_id: int) -> pa.Table:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        id_by_sample = dict(zip(sample_ids, range(min_id, min_id + len(sample_ids)), strict=True))
        dedup_ids = [id_by_sample[sample_id] for sample_id in table.column("sample_id").to_pylist()]
        return table.append_column(CURATOR_DEDUP_ID_STR, pa.array(dedup_ids, type=pa.int64()))

    def _generate_ids_func(self, filepath: str | list[str], table: pa.Table) -> pa.Table:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR in table.column_names:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {filepath}, not generating new IDs")
            return table

        sample_ids = self._sample_ids_for_ids(table)
        if not sample_ids:
            return self._append_sample_ids(table, sample_ids, 0)
        min_id = ray.get(self.id_generator.register_batch.remote(filepath, len(sample_ids)))
        return self._append_sample_ids(table, sample_ids, min_id)

    def _assign_ids_func(self, filepath: str | list[str], table: pa.Table) -> pa.Table:
        from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

        if CURATOR_DEDUP_ID_STR in table.column_names:
            logger.warning(f"Column {CURATOR_DEDUP_ID_STR} already exists in {filepath}, not re-assigning IDs")
            return table

        sample_ids = self._sample_ids_for_ids(table)
        if not sample_ids:
            return self._append_sample_ids(table, sample_ids, 0)
        min_id, max_id = ray.get(self.id_generator.get_batch_range.remote(filepath, None))
        expected_count = max_id - min_id + 1
        if expected_count != len(sample_ids):
            msg = (
                f"Cannot assign interleaved sample IDs for {filepath}: id generator registered "
                f"{expected_count} sample(s), but reader reconstructed {len(sample_ids)} sample_id(s). "
                "Use the same input file grouping as the identification run."
            )
            raise RuntimeError(msg)
        return self._append_sample_ids(table, sample_ids, min_id)

    def _apply_ids(self, filepath: str | list[str], table: pa.Table) -> pa.Table:
        if self._generate_ids:
            return self._generate_ids_func(filepath, table)
        if self._assign_ids:
            return self._assign_ids_func(filepath, table)
        return table

    @staticmethod
    def _source_files_for_split(
        split: pa.Table,
        idx: int,
        sample_id_to_path: dict[str, str],
        all_paths: list[str],
    ) -> list[str]:
        """Return source_files for one split, annotated with the split index for lineage tracking.

        The ``::split_NNN`` suffix is appended so that downstream consumers can correlate
        each output batch back to the exact split of its source file(s), even when a single
        source file is split into multiple batches by ``max_batch_bytes``.
        """
        seen: set[str] = set()
        for sid in split["sample_id"].unique().to_pylist():
            path = sample_id_to_path.get(sid)
            if path is not None:
                seen.add(path)
        contributing = [p for p in all_paths if p in seen]
        if not contributing:
            logger.warning(
                "_source_files_for_split: no source path found for any sample_id in this split "
                "(possible null sample_ids); falling back to all {} source path(s).",
                len(all_paths),
            )
            contributing = all_paths
        return [f"{p}::split_{idx:05d}" for p in contributing]

    def ray_stage_spec(self) -> dict[str, Any]:
        from nemo_curator.backends.utils import RayStageSpecKeys

        return {RayStageSpecKeys.IS_ACTOR_STAGE: self._generate_ids or self._assign_ids}
