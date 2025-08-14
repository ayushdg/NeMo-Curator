from itertools import pairwise
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask
from ray_curator.utils.file_utils import delete_dir, get_fs, is_not_empty


class BucketsToEdgesStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """
    Stage that takes in a file consiting of LSH bucket ids and document ids belonging to the bucket
    and outputs a file consisting of edges between documents with the same bucket id.

    Args:
        doc_id_column: The column name containing the document ids for each bucket.
        output_dir: The directory to write the output file to.
        read_storage_options: Storage options to pass to the fsspec compatible filesystem used to read the input file.
        write_storage_options: Storage options to pass to the fsspec compatible filesystem used to write the output file.
    """

    _name = "BucketsToEdgesStage"
    _resources = Resources(cpus=1.0)

    def __init__(
        self,
        output_dir: str,
        doc_id_column: str = CURATOR_DEDUP_ID_STR,
        read_storage_options: dict[str, Any] | None = None,
        write_storage_options: dict[str, Any] | None = None,
    ):
        self.doc_id_column = doc_id_column
        self.read_storage_options = read_storage_options if read_storage_options is not None else {}
        self.write_storage_options = write_storage_options if write_storage_options is not None else {}
        self.output_fs = get_fs(output_dir, write_storage_options)
        self.output_dir = self.output_fs.sep.join([output_dir, self.name])

        # Handle output directory cleanup logic
        if is_not_empty(self.output_dir, self.output_fs):
            logger.warning(f"Output directory {self.output_dir} is not empty. Deleting it.")
            delete_dir(self.output_dir, self.output_fs)
        self.output_fs.mkdirs(self.output_dir, exist_ok=True)

    def process(self, task: FileGroupTask) -> FileGroupTask:
        input_fs = get_fs(task.data[0], self.read_storage_options)
        df = pq.read_table(task.data, filesystem=input_fs)
        edges = []
        for bucket_docs in df[self.doc_id_column]:
            edges.extend(pairwise(bucket_docs))
        edges = [list(edge) for edge in edges]
        edges = pa.Table.from_pandas(
            pd.DataFrame(edges, columns=[f"{self.doc_id_column}_x", f"{self.doc_id_column}_y"])
        )

        output_path = self.output_fs.sep.join([self.output_dir, f"{task.task_id}.parquet"])
        pq.write_table(edges, output_path, filesystem=self.output_fs)
        return FileGroupTask(
            task_id=f"{task.task_id}",
            dataset_name=f"{task.dataset_name}_edges",
            data=[output_path],
            _metadata={**task._metadata, "storage_options": self.write_storage_options},
            _stage_perf=task._stage_perf,
        )
