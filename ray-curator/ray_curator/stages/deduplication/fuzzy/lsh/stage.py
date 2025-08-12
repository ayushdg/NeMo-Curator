from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ray_curator.backends.experimental.utils import RayStageSpecKeys
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.fuzzy.lsh.lsh import LSHActor
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask


@dataclass
class LSHStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that performs LSH on a FileGroupTask containing minhash data.

    The executor will process this stage in iterations based on bands_per_iteration.
    """

    _name = "LSHStage"
    _resources = Resources(gpus=1.0)

    # Core Algo objects
    actor_class = LSHActor

    # LSH parameters
    num_bands: int
    minhashes_per_band: int
    # Data parameters
    id_column: str = CURATOR_DEDUP_ID_STR
    minhash_field: str = "_minhash_signature"
    output_dir: str = "./"
    read_kwargs: dict[str, Any] | None = None
    write_kwargs: dict[str, Any] | None = None
    # Shuffle parameters
    rmm_pool_size: int = 1024 * 1024 * 1024
    spill_device: int | str | None = "auto"
    enable_statistics: bool = False
    bands_per_iteration: int = 5  # number of bands to process in each iteration

    def __post_init__(self):
        super().__init__()

        self.actor_kwargs = {
            "num_bands": self.num_bands,
            "minhashes_per_band": self.minhashes_per_band,
            "id_column": self.id_column,
            "minhash_field": self.minhash_field,
            "rmm_pool_size": self.rmm_pool_size,
            "spill_device": self.spill_device,
            "enable_statistics": self.enable_statistics,
            "read_kwargs": self.read_kwargs if self.read_kwargs is not None else {},
            "write_kwargs": self.write_kwargs if self.write_kwargs is not None else {},
        }

        if self.bands_per_iteration < 1 or self.bands_per_iteration > self.num_bands:
            err_msg = (
                f"Invalid bands_per_iteration: {self.bands_per_iteration}, must be in range [1, {self.num_bands}]"
            )
            raise ValueError(err_msg)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        err_msg = "LSHProcessingStage does not support the process method."
        raise NotImplementedError(err_msg)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_LSH_STAGE: True,
        }

    def _check_actor_obj(self) -> None:
        if not hasattr(self, "_actor_obj") or not isinstance(self._actor_obj, self.actor_class):
            error = "Actor object not initialized. This might be because an incorrect executor was used or it failed to setup the stage properly."
            raise RuntimeError(error)

    def read_and_insert(self, task: FileGroupTask, band_range: tuple[int, int]) -> FileGroupTask:
        self._check_actor_obj()
        result = self._actor_obj.read_and_insert(task.data, band_range)
        self.output_columns = result
        self.dataset_name = task.dataset_name
        return task

    def insert_finished(self) -> None:
        self._check_actor_obj()
        self._actor_obj.insert_finished()

    def extract_and_write(self) -> list[FileGroupTask]:
        self._check_actor_obj()
        partition_paths = self._actor_obj.extract_and_write()
        return [
            FileGroupTask(
                task_id=partition_id,
                dataset_name=self.dataset_name + f"{self.name}",
                data=path,
                _metadata={
                    "partition_index": partition_id,
                    "total_partitions": len(partition_paths),
                    "output_columns": self.output_columns,
                },
            )
            for partition_id, path in partition_paths
        ]

    def teardown(self) -> None:
        self._check_actor_obj()
        self._actor_obj.cleanup()

    def get_band_iterations(self) -> Iterator[tuple[int, int]]:
        """Get all band ranges for iteration."""
        for band_start in range(0, self.num_bands, self.bands_per_iteration):
            band_range = (band_start, min(band_start + self.bands_per_iteration, self.num_bands))
            yield band_range
