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

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional

import fsspec
from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowBase, WorkflowRunResult
from nemo_curator.stages.deduplication.fuzzy.duplicate_groups import ShuffleDuplicateGroupsStage
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR, IdGeneratorBase
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.deduplication.write_duplicates import ReverseIdIndex, WriteTextDuplicatesStage
from nemo_curator.tasks import EmptyTask, FileGroupTask

if TYPE_CHECKING:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor


@dataclass
class WriteTextDuplicatesWorkflow(WorkflowBase):
    """Write out the duplicate documents identified by fuzzy deduplication, grouped by group id.

    Two phases:
      1. Shuffle the connected-components ``(id, group)`` output by ``_duplicate_group_id`` into many
         partitions so every group's members land together (``ShuffleDuplicateGroupsStage``).
         Skipped if ``duplicate_groups_path`` already points at a shuffled output.
      2. For each shuffled partition, reverse-look-up the original input file groups that hold its
         ids (via a driver-built index over the id generator), re-read them, and GPU-merge to write
         the full duplicate documents annotated with ``_duplicate_group_id`` (``WriteTextDuplicatesStage``).

    The input file groups passed to ``run`` (or generated from ``input_path``) **must match the
    partitioning used during minhash/identification** so the id generator hashes line up and ids
    reproduce positionally.
    """

    # required
    output_path: str
    id_generator_path: str

    # source of the duplicate (id, group) list: provide the CC output dir to shuffle it here, or a
    # pre-shuffled FuzzyDuplicateGroups dir to skip the shuffle.
    connected_components_path: str | None = None
    duplicate_groups_path: str | None = None

    # original input dataset (used to build the reverse index and to re-read content)
    input_path: str | list[str] | None = None
    input_filetype: Literal["parquet", "jsonl"] = "parquet"
    input_files_per_partition: int | None = None
    input_blocksize: str | int | None = None
    input_file_extensions: list[str] | None = None
    input_fields: list[str] | None = None
    input_kwargs: dict[str, Any] | None = None

    # id / group fields
    id_field: str = CURATOR_DEDUP_ID_STR
    duplicate_group_field: str = CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
    id_generator_storage_options: dict[str, Any] | None = None

    # shuffle config
    shuffle_nparts: int | None = None
    shuffle_rmm_pool_size: int | Literal["auto"] | None = "auto"
    shuffle_spill_memory_limit: int | Literal["auto"] | None = "auto"
    connected_components_kwargs: dict[str, Any] | None = None

    # output config
    output_fields: list[str] | None = None
    output_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.connected_components_path is None and self.duplicate_groups_path is None:
            msg = "Either connected_components_path or duplicate_groups_path must be provided."
            raise ValueError(msg)

    def _input_file_groups(self, initial_tasks: list[FileGroupTask] | None) -> list[FileGroupTask]:
        if initial_tasks is not None:
            if self.input_path is not None:
                logger.warning("Ignoring input_path as initial_tasks (input file groups) are provided.")
            return initial_tasks
        if self.input_path is None:
            msg = "input_path is required to build the reverse index when initial_tasks are not provided."
            raise ValueError(msg)
        partitioner = FilePartitioningStage(
            file_paths=self.input_path,
            files_per_partition=self.input_files_per_partition,
            blocksize=self.input_blocksize,
            file_extensions=self.input_file_extensions,
            storage_options=(self.input_kwargs or {}).get("storage_options"),
        )
        return partitioner.process(EmptyTask)

    def _build_reverse_index(self, input_tasks: list[FileGroupTask]) -> ReverseIdIndex:
        storage_options = self.id_generator_storage_options or {}
        with fsspec.open(self.id_generator_path, mode="r", **storage_options) as f:
            data = json.load(f)
        generator = IdGeneratorBase(batch_registry=data["batch_registry"])

        ranges: list[tuple[int, int, list[str]]] = []
        missing = 0
        for task in input_tasks:
            files = list(task.data)
            try:
                min_id, max_id = generator.get_batch_range(files=files, key=None)
            except KeyError:
                missing += 1
                continue
            ranges.append((int(min_id), int(max_id), files))
        if missing:
            logger.warning(
                f"{missing} input file group(s) were not found in the id generator registry at "
                f"{self.id_generator_path}. Ensure the input file groups match the partitioning used "
                f"during minhash/identification."
            )
        if not ranges:
            msg = (
                "No input file groups matched the id generator registry; cannot build the reverse "
                "index. The input partitioning likely differs from the minhash run."
            )
            raise ValueError(msg)
        return ReverseIdIndex(ranges)

    def _shuffle_pipeline(self) -> Pipeline:
        cc_kwargs = self.connected_components_kwargs
        return Pipeline(
            name="shuffle_duplicate_groups_pipeline",
            stages=[
                FilePartitioningStage(
                    file_paths=self.connected_components_path,
                    file_extensions=[".parquet"],
                    storage_options=(cc_kwargs or {}).get("storage_options"),
                ),
                ShuffleDuplicateGroupsStage(
                    duplicate_group_field=self.duplicate_group_field,
                    document_id_field=self.id_field,
                    total_nparts=self.shuffle_nparts,
                    output_path=self.output_path,
                    read_kwargs=cc_kwargs,
                    write_kwargs=cc_kwargs,
                    rmm_pool_size=self.shuffle_rmm_pool_size,
                    spill_memory_limit=self.shuffle_spill_memory_limit,
                ),
            ],
        )

    def _duplicate_group_tasks_from_path(self) -> list[FileGroupTask]:
        partitioner = FilePartitioningStage(
            file_paths=self.duplicate_groups_path,
            files_per_partition=1,
            file_extensions=[".parquet"],
            storage_options=(self.connected_components_kwargs or {}).get("storage_options"),
        )
        return partitioner.process(EmptyTask)

    def run(
        self,
        initial_tasks: list[FileGroupTask] | None = None,
        executor: Optional["RayActorPoolExecutor"] = None,
    ) -> WorkflowRunResult:
        """Run the workflow.

        Args:
            initial_tasks: The input file groups used during minhash/identification (same
                partitioning). If None, they are generated from ``input_path``.
            executor: Executor to use. Defaults to ``RayActorPoolExecutor`` (required by the shuffle).
        """
        from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor

        if executor is None:
            executor = RayActorPoolExecutor()

        workflow_result = WorkflowRunResult(workflow_name="write_text_duplicates")
        start_time = time.time()

        # 1) Build the reverse index on the driver (local; no id generator actor needed).
        input_tasks = self._input_file_groups(initial_tasks)
        reverse_index = self._build_reverse_index(input_tasks)
        logger.info(f"Built reverse index over {len(reverse_index)} input file groups")

        # 2) Obtain the shuffled (id, group) partitions.
        if self.duplicate_groups_path is not None:
            duplicate_group_tasks = self._duplicate_group_tasks_from_path()
        else:
            shuffle_tasks = self._shuffle_pipeline().run(executor=executor, initial_tasks=None)
            duplicate_group_tasks = shuffle_tasks or []
        workflow_result.add_pipeline_tasks("shuffle_duplicate_groups", duplicate_group_tasks)

        if not duplicate_group_tasks:
            logger.info("No duplicate groups found. Nothing to write.")
            workflow_result.add_metadata("num_duplicates_written", 0)
            workflow_result.add_metadata("total_time", time.time() - start_time)
            return workflow_result

        # 3) Reverse-lookup + merge + write the duplicate documents.
        write_stage = WriteTextDuplicatesStage(
            reverse_index=reverse_index,
            output_path=self.output_path,
            input_filetype=self.input_filetype,
            id_field=self.id_field,
            duplicate_group_field=self.duplicate_group_field,
            fields=self.input_fields,
            output_fields=self.output_fields,
            read_kwargs=self.input_kwargs,
            write_kwargs=self.output_kwargs,
            duplicate_read_kwargs=self.connected_components_kwargs,
        )
        write_pipeline = Pipeline(name="write_text_duplicates_pipeline", stages=[write_stage])
        output_tasks = write_pipeline.run(executor=executor, initial_tasks=duplicate_group_tasks)

        num_written = sum((t._metadata or {}).get("num_duplicates_written", 0) for t in (output_tasks or []))
        workflow_result.add_pipeline_tasks("write_duplicates", output_tasks)
        workflow_result.add_metadata("num_duplicates_written", num_written)
        workflow_result.add_metadata("total_time", time.time() - start_time)
        logger.info(f"Wrote {num_written} duplicate documents in {time.time() - start_time:.2f}s")
        return workflow_result
