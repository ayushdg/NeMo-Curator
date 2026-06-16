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

"""Driver-orchestrated workflow that writes duplicate documents grouped by ``_duplicate_group_id``
using the disk-based external-sort / range-merge stages (no collective shuffle, no reduce primitive).

Phases (each a map-only Curator pipeline; the glue runs on the driver):
  1. sort the connected-components ``(id, group)`` output by id (for pushdown joins)
  2. materialize one group-sorted run per map task (input read once)
  3. driver: collect run paths + compute ``group_id`` quantile cut-points
  4. merge each ``group_id`` range across runs into B group-complete output files
"""

import json
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import fsspec
import numpy as np
import pyarrow.dataset as pa_ds
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowBase, WorkflowRunResult
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.grouped_duplicates import (
    SORTED_CC_SUBDIR,
    MaterializeDuplicateRunsStage,
    MergeDuplicateGroupsStage,
    SortDuplicatesByIdStage,
)
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_fs

if TYPE_CHECKING:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor


@dataclass
class WriteDuplicateGroupsWorkflow(WorkflowBase):
    """Write duplicate documents grouped by ``_duplicate_group_id`` to B group-complete files.

    The input file groups must match the partitioning used during minhash/identification so the id
    generator hashes line up and positional ids reproduce.
    """

    # required
    output_path: str
    id_generator_path: str
    connected_components_path: str

    # original input dataset
    input_path: str | list[str] | None = None
    input_filetype: Literal["parquet", "jsonl"] = "parquet"
    input_blocksize: str | int | None = "1.5GiB"
    input_files_per_partition: int | None = None
    input_file_extensions: list[str] | None = None
    input_fields: list[str] | None = None
    input_kwargs: dict[str, Any] | None = None

    # fields
    id_field: str = CURATOR_DEDUP_ID_STR
    group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
    output_fields: list[str] | None = None
    id_generator_storage_options: dict[str, Any] | None = None

    # parallelism knobs
    n_map: int = 128  # number of run files (map tasks)
    num_output_partitions: int = 256  # B: final group-complete files
    row_group_size: int = 1_000_000

    # io
    cc_kwargs: dict[str, Any] | None = None  # reading/writing CC + sorted CC + runs
    output_kwargs: dict[str, Any] | None = None

    def _num_input_groups(self) -> int:
        """Number of input file groups minhash registered (= size of the id generator registry)."""
        storage_options = self.id_generator_storage_options or {}
        with fsspec.open(self.id_generator_path, mode="r", **storage_options) as f:
            data = json.load(f)
        return len(data["batch_registry"])

    def _compute_group_cutpoints(self, sorted_cc_path: str) -> list[int]:
        """Quantile cut-points on group_id so the B merge ranges are ~row-balanced (CPU, no GPU)."""
        storage_options = (self.cc_kwargs or {}).get("storage_options")
        fs = get_fs(sorted_cc_path, storage_options)
        try:
            import pyarrow.fs as pa_fs

            pa_filesystem = pa_fs.PyFileSystem(pa_fs.FSSpecHandler(fs)) if storage_options else None
        except Exception:  # noqa: BLE001
            pa_filesystem = None
        dataset = pa_ds.dataset(sorted_cc_path, format="parquet", filesystem=pa_filesystem)
        col = dataset.to_table(columns=[self.group_field]).column(0).to_numpy()
        probs = np.linspace(0.0, 1.0, self.num_output_partitions + 1)[1:-1]
        cuts = np.quantile(col, probs, method="lower").astype("int64")
        return np.unique(cuts).tolist()

    def _build_merge_tasks(self, run_paths: list[str], cuts: list[int]) -> list[FileGroupTask]:
        bounds: list[int | None] = [None, *cuts, None]
        tasks = []
        for b in range(len(cuts) + 1):
            tasks.append(
                FileGroupTask(
                    task_id=f"merge_{b}",
                    dataset_name="duplicate_groups",
                    data=list(run_paths),
                    _metadata={"group_lo": bounds[b], "group_hi": bounds[b + 1], "partition_index": b},
                )
            )
        return tasks

    def run(
        self,
        initial_tasks: list[FileGroupTask] | None = None,
        executor: Optional["RayActorPoolExecutor"] = None,
    ) -> WorkflowRunResult:
        from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
        from nemo_curator.stages.deduplication.id_generator import (
            create_id_generator_actor,
            kill_id_generator_actor,
        )

        if executor is None:
            executor = RayActorPoolExecutor()

        result = WorkflowRunResult(workflow_name="write_duplicate_groups")
        t0 = time.time()

        # batch_size coarsens input groups -> runs: n_map = ceil(num_groups / batch_size).
        num_groups = self._num_input_groups()
        batch_size = max(1, math.ceil(num_groups / self.n_map))
        logger.info(f"{num_groups} input groups; batch_size={batch_size} -> ~{math.ceil(num_groups / batch_size)} runs")

        # MaterializeDuplicateRunsStage looks up id ranges from the IdGenerator actor, so load it.
        create_id_generator_actor(self.id_generator_path, storage_options=self.id_generator_storage_options)
        out_tasks: list[FileGroupTask] | None = None
        run_paths: list[str] = []
        merge_tasks: list[FileGroupTask] = []
        try:
            # 1) sort CC by id (map)
            sort_pipeline = Pipeline(
                name="sort_cc_by_id_pipeline",
                stages=[
                    FilePartitioningStage(
                        file_paths=self.connected_components_path,
                        file_extensions=[".parquet"],
                        files_per_partition=1,
                        storage_options=(self.cc_kwargs or {}).get("storage_options"),
                    ),
                    SortDuplicatesByIdStage(
                        output_path=self.output_path,
                        id_field=self.id_field,
                        group_field=self.group_field,
                        read_kwargs=self.cc_kwargs,
                        write_kwargs=self.cc_kwargs,
                        row_group_size=self.row_group_size,
                    ),
                ],
            )
            result.add_pipeline_tasks("sort_cc", sort_pipeline.run(executor=executor, initial_tasks=None))
            sorted_cc_path = get_fs(self.output_path, (self.cc_kwargs or {}).get("storage_options")).sep.join(
                [self.output_path, SORTED_CC_SUBDIR]
            )

            # 2) materialize group-sorted runs (input read once; batch_size input groups -> 1 run)
            materialize_stage = MaterializeDuplicateRunsStage(
                sorted_cc_path=sorted_cc_path,
                output_path=self.output_path,
                input_filetype=self.input_filetype,
                id_field=self.id_field,
                group_field=self.group_field,
                fields=self.input_fields,
                output_fields=self.output_fields,
                read_kwargs=self.input_kwargs,
                write_kwargs=self.output_kwargs,
                cc_read_kwargs=self.cc_kwargs,
                num_output_partitions=self.num_output_partitions,
            ).with_(batch_size=batch_size)

            if initial_tasks is not None:
                if self.input_path is not None:
                    logger.warning("Ignoring input_path; using provided initial_tasks as input file groups.")
                materialize_stages = [materialize_stage]
                materialize_initial = initial_tasks
            elif self.input_path is None:
                msg = "input_path is required when initial_tasks are not provided."
                raise ValueError(msg)
            else:
                materialize_stages = [
                    FilePartitioningStage(
                        file_paths=self.input_path,
                        files_per_partition=self.input_files_per_partition,
                        blocksize=self.input_blocksize,
                        file_extensions=self.input_file_extensions,
                        storage_options=(self.input_kwargs or {}).get("storage_options"),
                    ),
                    materialize_stage,
                ]
                materialize_initial = None

            run_tasks = Pipeline(name="materialize_duplicate_runs_pipeline", stages=materialize_stages).run(
                executor=executor, initial_tasks=materialize_initial
            )
            run_paths = [p for t in (run_tasks or []) for p in t.data]
            result.add_pipeline_tasks("materialize_runs", run_tasks)
            logger.info(f"Materialized {len(run_paths)} runs")
            if not run_paths:
                logger.warning("No duplicate runs produced; nothing to merge.")
                result.add_metadata("total_time", time.time() - t0)
                return result

            # 3) driver: group_id quantile cut-points (CPU)
            cuts = self._compute_group_cutpoints(sorted_cc_path)
            logger.info(f"Computed {len(cuts)} group_id cut-points -> {len(cuts) + 1} output partitions")

            # 4) merge each group_id range across runs (map)
            merge_tasks = self._build_merge_tasks(run_paths, cuts)
            out_tasks = Pipeline(
                name="merge_duplicate_groups_pipeline",
                stages=[
                    MergeDuplicateGroupsStage(
                        output_path=self.output_path,
                        id_field=self.id_field,
                        group_field=self.group_field,
                        storage_options=(self.output_kwargs or {}).get("storage_options"),
                        row_group_size=self.row_group_size,
                    ),
                ],
            ).run(executor=executor, initial_tasks=merge_tasks)
            result.add_pipeline_tasks("merge_groups", out_tasks)
        finally:
            kill_id_generator_actor()

        num_rows = sum((t._metadata or {}).get("num_rows", 0) for t in (out_tasks or []))
        result.add_metadata("num_duplicate_docs_written", num_rows)
        result.add_metadata("num_runs", len(run_paths))
        result.add_metadata("num_output_partitions", len(merge_tasks))
        result.add_metadata("total_time", time.time() - t0)
        logger.success(f"Wrote {num_rows} grouped duplicate docs across {len(merge_tasks)} files in {time.time()-t0:.1f}s")
        return result
