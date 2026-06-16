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
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowBase, WorkflowRunResult
from nemo_curator.stages.deduplication.fuzzy.identify_duplicates import DUPLICATE_IDS_SUBDIR
from nemo_curator.stages.deduplication.fuzzy.minhash import InterleavedTextMode
from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.deduplication.removal_workflow import InterleavedDuplicatesRemovalWorkflow
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import get_fs


@dataclass
class InterleavedTextFuzzyDeduplicationWorkflow(WorkflowBase):
    """End-to-end fuzzy deduplication for interleaved text-bearing samples."""

    cache_path: str
    output_path: str
    interleaved_text_mode: InterleavedTextMode

    input_path: str | list[str] | None = None
    input_blocksize: str | int | None = "1GiB"
    input_files_per_partition: int | None = None
    input_file_extensions: list[str] = field(default_factory=lambda: [".parquet"])
    input_task_limit: int | None = None
    read_kwargs: dict[str, Any] | None = None
    cache_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None

    text_field: str = "text"
    metadata_json_path: str | None = "$.content"
    perform_removal: bool = True
    deduplicated_output_path: str | None = None
    drop_id_field: bool = True

    seed: int = 42
    char_ngrams: int = 24
    num_bands: int = 20
    minhashes_per_band: int = 13
    use_64_bit_hash: bool = False
    bands_per_iteration: int = 5
    lsh_num_output_partitions: int | None = None
    lsh_rmm_pool_size: int | str | None = "auto"
    lsh_spill_memory_limit: int | str | None = "auto"
    env_vars: dict[str, Any] | None = None

    def _deduplicated_output_path(self) -> str:
        if self.deduplicated_output_path is not None:
            return self.deduplicated_output_path
        output_fs = get_fs(
            self.output_path,
            self.write_kwargs.get("storage_options") if self.write_kwargs is not None else None,
        )
        return output_fs.sep.join([self.output_path, "deduplicated"])

    def _duplicate_ids_path(self) -> str:
        output_fs = get_fs(
            self.output_path,
            self.write_kwargs.get("storage_options") if self.write_kwargs is not None else None,
        )
        return output_fs.sep.join([self.output_path, DUPLICATE_IDS_SUBDIR])

    def _create_input_filegroups_pipeline(self) -> Pipeline:
        if self.input_path is None:
            msg = "input_path is required when initial_tasks is None"
            raise ValueError(msg)
        return Pipeline(
            name="input_filegroups_pipeline",
            stages=[
                FilePartitioningStage(
                    file_paths=self.input_path,
                    files_per_partition=self.input_files_per_partition,
                    blocksize=self.input_blocksize,
                    file_extensions=self.input_file_extensions,
                    storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs is not None else None,
                    limit=self.input_task_limit,
                )
            ],
        )

    def _create_identification_workflow(self) -> FuzzyDeduplicationWorkflow:
        return FuzzyDeduplicationWorkflow(
            cache_path=self.cache_path,
            output_path=self.output_path,
            input_path=None,
            input_filetype="parquet",
            input_blocksize=self.input_blocksize or "1GiB",
            input_file_extensions=self.input_file_extensions,
            read_kwargs=self.read_kwargs,
            cache_kwargs=self.cache_kwargs,
            write_kwargs=self.write_kwargs,
            input_dataset_type="interleaved",
            interleaved_text_mode=self.interleaved_text_mode,
            interleaved_metadata_json_path=self.metadata_json_path,
            text_field=self.text_field,
            perform_removal=False,
            seed=self.seed,
            char_ngrams=self.char_ngrams,
            num_bands=self.num_bands,
            minhashes_per_band=self.minhashes_per_band,
            use_64_bit_hash=self.use_64_bit_hash,
            bands_per_iteration=self.bands_per_iteration,
            lsh_num_output_partitions=self.lsh_num_output_partitions,
            lsh_rmm_pool_size=self.lsh_rmm_pool_size,
            lsh_spill_memory_limit=self.lsh_spill_memory_limit,
            env_vars=self.env_vars,
        )

    def _create_removal_workflow(self, id_generator_path: str) -> InterleavedDuplicatesRemovalWorkflow:
        return InterleavedDuplicatesRemovalWorkflow(
            input_path=None,
            ids_to_remove_path=self._duplicate_ids_path(),
            output_path=self._deduplicated_output_path(),
            input_kwargs=self.read_kwargs,
            duplicate_id_read_kwargs=self.write_kwargs,
            id_generator_path=id_generator_path,
            id_generator_storage_options=self.write_kwargs.get("storage_options") if self.write_kwargs else None,
            drop_id_field=self.drop_id_field,
            output_kwargs=self.write_kwargs,
            output_mode="ignore",
            materialize_on_write=False,
        )

    @staticmethod
    def _create_removal_executor(executor: RayActorPoolExecutor) -> RayDataExecutor:
        return RayDataExecutor(config=executor.config, ignore_head_node=executor.ignore_head_node)

    def _resolve_initial_tasks(
        self,
        workflow_result: WorkflowRunResult,
        executor: RayActorPoolExecutor,
        initial_tasks: list[FileGroupTask] | None,
    ) -> list[FileGroupTask]:
        if initial_tasks is not None:
            if self.input_task_limit is not None and len(initial_tasks) > self.input_task_limit:
                logger.warning(
                    f"Initial tasks provided ({len(initial_tasks)}) is greater than input_task_limit ({self.input_task_limit}), truncating to {self.input_task_limit}"
                )
                return initial_tasks[: self.input_task_limit]
            return initial_tasks

        input_filegroups_pipeline = self._create_input_filegroups_pipeline()
        input_start_time = time.time()
        resolved_tasks = input_filegroups_pipeline.run(executor=executor, initial_tasks=None)
        input_filegroups_time = time.time() - input_start_time
        workflow_result.add_metadata("input_filegroups_time", input_filegroups_time)
        workflow_result.add_pipeline_tasks("input_filegroups", resolved_tasks)
        logger.info(f"Created input tasks from {self.input_path} in {input_filegroups_time:.2f} seconds")
        return resolved_tasks or []

    def run(
        self, initial_tasks: list[FileGroupTask] | None = None, executor: RayActorPoolExecutor | None = None
    ) -> WorkflowRunResult:
        workflow_result = WorkflowRunResult(workflow_name="interleaved_text_fuzzy_deduplication")
        total_start_time = time.time()

        if executor is None:
            executor = RayActorPoolExecutor()

        input_tasks = self._resolve_initial_tasks(workflow_result, executor, initial_tasks)

        identification_workflow = self._create_identification_workflow()
        identification_result = identification_workflow.run(initial_tasks=input_tasks, executor=executor)
        for pipeline_name, tasks in identification_result.pipeline_tasks.items():
            workflow_result.add_pipeline_tasks(pipeline_name, tasks)
        workflow_result.extend_metadata(identification_result.metadata)

        if self.perform_removal:
            id_generator_path = identification_result.get_metadata("id_generator_path")
            if id_generator_path is None:
                msg = "Fuzzy identification did not report an id_generator_path required for removal"
                raise RuntimeError(msg)
            removal_workflow = self._create_removal_workflow(id_generator_path)
            removal_start_time = time.time()
            removal_executor = self._create_removal_executor(executor)
            logger.info("Running interleaved duplicate removal with RayDataExecutor")
            removal_result = removal_workflow.run(executor=removal_executor, initial_tasks=input_tasks)
            removal_time = time.time() - removal_start_time
            workflow_result.add_pipeline_tasks("removal", removal_result.pipeline_tasks.get("removal", []))
            workflow_result.add_metadata("removal_time", removal_time)
            workflow_result.add_metadata(
                "num_duplicates_removed", removal_result.get_metadata("num_duplicates_removed")
            )
            workflow_result.add_metadata("deduplicated_output_path", self._deduplicated_output_path())

        workflow_result.add_metadata("total_time", time.time() - total_start_time)
        return workflow_result
