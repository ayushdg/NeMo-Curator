import os
from typing import TYPE_CHECKING, Any

import cudf
import cugraph
from cugraph.dask.comms.comms_wrapper import init_subcomms as c_init_subcomms
from pylibcugraph import GraphProperties, MGGraph, ResourceHandle
from pylibcugraph import weakly_connected_components as pylibcugraph_wcc

from ray_curator.backends.experimental.utils import RayStageSpecKeys
from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
    CURATOR_ID_GENERATOR_ACTOR_NAME,
    IdGenerator,
)
from ray_curator.stages.deduplication.io_utils import DeduplicationIO
from ray_curator.stages.resources import Resources
from ray_curator.tasks.file_group import FileGroupTask

if TYPE_CHECKING:
    from ray_curator.backends.base import WorkerMetadata


class ConnectedComponentsStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    def __init__(
        self,
        output_dir: str,
        source_column: str = f"{CURATOR_DEDUP_ID_STR}_x",
        destination_column: str = f"{CURATOR_DEDUP_ID_STR}_y",
        read_kwargs: dict | None = None,
        write_kwargs: dict | None = None,
    ):
        # Initialize parent classes
        ProcessingStage.__init__(self)
        DeduplicationIO.__init__(self, id_generator=None)

        self.output_dir = output_dir
        self.source_column = source_column
        self.destination_column = destination_column
        self.read_storage_options = read_kwargs.get("storage_options", {}) if read_kwargs is not None else {}
        self.write_storage_options = write_kwargs.get("storage_options", {}) if write_kwargs is not None else {}

        self._name = self.__class__.__name__
        self._resources = Resources(cpus=1.0, gpus=1.0)

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self.id_generator = IdGenerator.options(
            name=CURATOR_ID_GENERATOR_ACTOR_NAME, get_if_exists=True, lifetime="detached"
        ).remote()
        self._setup_post()

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            RayStageSpecKeys.IS_RAFT_ACTOR: True,
        }

    def _setup_post(self) -> None:
        """Setup the sub-communicator for cuGraph communications.

        This method is specific to cuGraph comms and is used to initialize the
        sub-communicator.
        """
        print("     Setting up cuGraph-subcom...", flush=True)
        c_init_subcomms(self._raft_handle, 1)

    def weakly_connected_components(self, df: cudf.DataFrame, src_col: str, dst_col: str) -> None:
        """Compute the weakly connected components of a graph.

        This method loads a chunk of the graph, creates a cuGraph object, and
        computes the weakly connected components using the MGGraph library.

        Parameters
        ----------
        start: int
            The start index of the chunk.
        stop: int
            The stop index of the chunk.
        """

        dg = cugraph.Graph(directed=False)

        src_array = df[src_col]
        dst_array = df[dst_col]

        rhandle = ResourceHandle(self._raft_handle.getHandle())

        graph_props = GraphProperties(
            is_multigraph=False,  # dg.properties.multi_edge (what is multi_edge)
            is_symmetric=not dg.graph_properties.directed,
        )
        print("Running graph creation")
        plc_graph = MGGraph(
            resource_handle=rhandle,
            graph_properties=graph_props,
            src_array=[src_array],
            dst_array=[dst_array],
            edge_id_array=None,
            edge_type_array=None,
            num_arrays=1,
            store_transposed=False,
            symmetrize=False,
            do_expensive_check=False,
            drop_multi_edges=True,
        )
        print("Running weakly connected components")
        res = pylibcugraph_wcc(
            resource_handle=rhandle,
            graph=plc_graph,
            offsets=None,
            indices=None,
            weights=None,
            labels=None,
            do_expensive_check=False,
        )
        print("Computing weakly connected components completed successfully!", flush=True)
        return res

    def process(self, task: FileGroupTask) -> FileGroupTask:
        err_msg = "ConnectedComponentsStage only support process batch"
        raise NotImplementedError(err_msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[FileGroupTask]:
        """
        Process an input file, compute minhashes, and write results to an output file.
        Automatically adds a unique _curator_id field to each document if not present.

        Parameters
        ----------
        infiles: str, list[str]
            Path to input file (JSONL format) or list of paths
        outfile: str
            Path to output file (Parquet format)
        columns: list, optional
            Columns to read from input file
        """
        input_files = []
        for task in tasks:
            input_files.extend(task.data)
        output_file = os.path.join(self.output_dir, self.name, f"{tasks[0].task_id}.parquet")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        edgelist_columns = [self.source_column, self.destination_column]
        df = self.read_parquet(input_files, columns=edgelist_columns)
        df = df.drop_duplicates(subset=edgelist_columns, ignore_index=True)
        vertices, labels = self.weakly_connected_components(df, self.source_column, self.destination_column)
        df = cudf.DataFrame()
        df[CURATOR_DEDUP_ID_STR] = vertices
        df["_duplicate_group_id"] = labels
        df.to_parquet(output_file, index=False)
        return [
            FileGroupTask(
                dataset_name=tasks[0].dataset_name,
                task_id=tasks[0].task_id,
                data=[output_file],
            )
        ]
