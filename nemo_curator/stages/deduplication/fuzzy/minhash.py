# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import cudf
import numpy as np
import rmm
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_DEFAULT_MINHASH_FIELD
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR, get_id_generator_actor
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.interleaved.utils.deduplication import sample_ordering
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.file_utils import create_or_overwrite_dir, get_fs

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

InterleavedTextMode = Literal["metadata_content", "text_rows"]


class MinHash(ABC):
    """
    Base class for computing minhash signatures of a document corpus
    """

    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
    ):
        """
        Parameters
        ----------
        seed: Seed for minhash permutations
        num_hashes: Length of minhash signature (No. of minhash permutations)
        char_ngrams: Width of text window (in characters) while computing minhashes.
        use_64bit_hash: Whether to use a 64 bit hash function.
        """
        self.num_hashes = num_hashes
        self.char_ngram = char_ngrams
        self.seed = seed
        self.use_64bit_hash = use_64bit_hash

    def generate_seeds(self, n_permutations: int = 260, seed: int = 0, bit_width: int = 32) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        This is a placeholder that child classes should implement if needed.
        """
        msg = "Child classes should implement this method if needed"
        raise NotImplementedError(msg)

    @abstractmethod
    def compute_minhashes(self, text_series: Any) -> Any:  # noqa: ANN401
        """
        Compute minhash signatures for the given dataframe text column.
        """


class GPUMinHash(MinHash):
    def __init__(
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
        pool: bool = False,
    ):
        # Initialize parent class
        MinHash.__init__(
            self,
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
        )

        # Initialize memory pool for cuDF
        if pool:
            rmm.reinitialize(pool_allocator=pool)

        # Generate seeds
        self.seeds = self.generate_seeds(
            n_permutations=self.num_hashes,
            seed=self.seed,
            bit_width=64 if self.use_64bit_hash else 32,
        )

    def generate_seeds(self, n_permutations: int = 260, seed: int = 0, bit_width: int = 32) -> np.ndarray:
        """
        Generate seeds for all minhash permutations based on the given seed.
        """
        gen = np.random.RandomState(seed)

        if bit_width == 32:  # noqa: PLR2004
            MERSENNE_PRIME = np.uint32((1 << 31) - 1)  # noqa: N806
            dtype = np.uint32
        elif bit_width == 64:  # noqa: PLR2004
            # For 64-bit, use a larger prime number suitable for 64-bit operations
            MERSENNE_PRIME = np.uint64((1 << 61) - 1)  # noqa: N806
            dtype = np.uint64
        else:
            msg = "Unsupported bit width. Use either 32 or 64."
            raise ValueError(msg)

        return np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=dtype),
                    gen.randint(0, MERSENNE_PRIME, dtype=dtype),
                )
                for _ in range(n_permutations)
            ],
            dtype=dtype,
        )

    def minhash32(self, ser: cudf.Series) -> cudf.Series:
        """
        Compute 32bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        seeds_a = cudf.Series(self.seeds[:, 0], dtype="uint32")
        seeds_b = cudf.Series(self.seeds[:, 1], dtype="uint32")

        return ser.str.minhash(a=seeds_a, b=seeds_b, seed=self.seeds[0][0], width=self.char_ngram)

    def minhash64(self, ser: cudf.Series) -> cudf.Series:
        """
        Compute 64bit minhashes based on the MurmurHash3 algorithm
        """
        if not isinstance(ser, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        seeds_a = cudf.Series(self.seeds[:, 0], dtype="uint64")
        seeds_b = cudf.Series(self.seeds[:, 1], dtype="uint64")

        return ser.str.minhash64(a=seeds_a, b=seeds_b, seed=self.seeds[0][0], width=self.char_ngram)

    def compute_minhashes(self, text_series: cudf.Series) -> cudf.Series:
        """
        Compute minhash signatures for the given text series.

        Parameters
        ----------
        text_series: cudf.Series
            Series containing text data to compute minhashes for

        Returns
        -------
        cudf.Series containing minhash signatures
        """
        if not isinstance(text_series, cudf.Series):
            msg = "Expected data of type cudf.Series"
            raise TypeError(msg)

        # Compute minhashes
        minhash_method = self.minhash64 if self.use_64bit_hash else self.minhash32
        return minhash_method(text_series)


class MinHashStage(ProcessingStage[FileGroupTask, FileGroupTask], DeduplicationIO):
    """
    ProcessingStage for computing MinHash signatures on documents for fuzzy deduplication.

    This stage takes FileGroupTask containing paths to input documents and produces
    FileGroupTask containing paths to computed minhash signature files. It uses GPU-accelerated
    MinHash computation to generate locality-sensitive hash signatures that can be used
    for approximate duplicate detection.

    The stage automatically handles:
    - Reading input files (JSONL or Parquet format)
    - Assigning unique Integer IDs to documents using the IdGenerator actor
    - Computing MinHash signatures using GPU acceleration
    - Writing results to Parquet files

    Parameters
    ----------
    output_path : str
        Base path where minhash output files will be written
    text_field : str, default="text"
        Name of the field containing text to compute minhashes from
    minhash_field : str, default="_minhash_signature"
        Name of the field where minhash signatures will be stored
    char_ngrams : int, default=24
        Width of character n-grams for minhashing
    num_hashes : int, default=260
        Number of hash functions (length of minhash signature)
    seed : int, default=42
        Random seed for reproducible minhash generation
    use_64bit_hash : bool, default=False
        Whether to use 64-bit hash functions (vs 32-bit)
    read_format : Literal["jsonl", "parquet"], default="jsonl"
        Format of input files
    read_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for reading input files
    write_kwargs : dict[str, Any] | None, default=None
        Additional keyword arguments for writing output files

    Examples
    --------
    >>> stage = MinHashStage(
    ...     output_path="/path/to/minhash/output",
    ...     text_field="content",
    ...     num_hashes=128,
    ...     char_ngrams=5
    ... )
    >>> # Use in a pipeline to process document batches
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        text_field: str = "text",
        minhash_field: str = CURATOR_DEFAULT_MINHASH_FIELD,
        char_ngrams: int = 24,
        num_hashes: int = 260,
        seed: int = 42,
        use_64bit_hash: bool = False,
        read_format: Literal["jsonl", "parquet"] = "jsonl",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        pool: bool = True,
    ):
        # Set ProcessingStage attributes
        self.name = self.__class__.__name__
        self.resources = Resources(gpus=1.0)  # Requires 1 GPU

        self.text_field = text_field
        self.minhash_field = minhash_field
        self.char_ngrams = char_ngrams
        self.num_hashes = num_hashes
        self.seed = seed
        self.use_64bit_hash = use_64bit_hash
        self.read_format = read_format
        self.read_kwargs = read_kwargs or {}
        self.write_kwargs = write_kwargs or {}
        self.pool = pool
        # Initialize the minhash processor in setup
        self.minhash_processor = None
        self.id_generator = None

        self.output_fs = get_fs(output_path, self.write_kwargs.get("storage_options", {}))
        self.output_path = self.output_fs.sep.join([output_path, self.name])
        create_or_overwrite_dir(self.output_path, storage_options=self.write_kwargs.get("storage_options", {}))

    def setup(self, _worker_metadata: "WorkerMetadata | None" = None) -> None:
        """Initialize the GPU MinHash processor and ID generator."""
        # Initialize the ID generator (will be shared across workers)

        try:
            self.id_generator = get_id_generator_actor()
        except ValueError as e:
            err_msg = """
            Failed to get ID generator actor. Start an ID generator actor via `create_id_generator_actor` if calling this stage directly.
            If using the FuzzyDedup API this should be started automatically.
            """
            raise ValueError(err_msg) from e

        # Initialize the GPU minhash processor
        self.minhash_processor = GPUMinHash(
            seed=self.seed,
            num_hashes=self.num_hashes,
            char_ngrams=self.char_ngrams,
            use_64bit_hash=self.use_64bit_hash,
            pool=self.pool,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define input requirements."""
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define outputs - produces FileGroupTask with minhash files."""
        return (["data"], [])

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """
        Process a group of files to compute minhashes.

        Args:
            task: FileGroupTask containing file paths to process

        Returns:
            FileGroupTask containing paths to minhash output files
        """

        if self.minhash_processor is None or self.id_generator is None:
            msg = "MinHash processor or ID generator not initialized. Call setup() first."
            raise RuntimeError(msg)

        output_file = self.output_fs.sep.join([self.output_path, f"{task._uuid}.parquet"])

        read_kwargs = self.read_kwargs.copy()

        # Read input file based on format
        if self.read_format == "jsonl":
            df = self.read_jsonl(filepath=task.data, columns=[self.text_field], assign_id=True, **read_kwargs)
        elif self.read_format == "parquet":
            df = self.read_parquet(filepath=task.data, columns=[self.text_field], assign_id=True, **read_kwargs)
        else:
            msg = f"Unsupported read format: {self.read_format}"
            raise ValueError(msg)

        result_df = df[[CURATOR_DEDUP_ID_STR]]
        result_df[self.minhash_field] = self.minhash_processor.compute_minhashes(df[self.text_field])

        # Write output file
        self.write_parquet(df=result_df, filepath=output_file, **self.write_kwargs)

        # Return FileGroupTask with output file
        return FileGroupTask(
            task_id=f"{task.task_id}",
            dataset_name=f"{task.dataset_name}_minhash",
            data=[output_file],
            _metadata={
                **task._metadata,
                "minhash_field": self.minhash_field,
                "num_hashes": self.num_hashes,
                "storage_options": self.write_kwargs.get("storage_options"),
            },
            _stage_perf=task._stage_perf,
        )


class InterleavedMinHashStage(MinHashStage):
    """
    ProcessingStage for computing sample-level MinHash signatures from interleaved multimodal Parquet data.

    This stage takes FileGroupTask input pointing to row-wise interleaved Parquet files, normalizes the rows to
    one document per sample_id, assigns one deduplication ID per sample, and writes the same minhash parquet schema
    produced by MinHashStage.
    """

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        text_mode: InterleavedTextMode,
        text_field: str = "text",
        minhash_field: str = CURATOR_DEFAULT_MINHASH_FIELD,
        char_ngrams: int = 24,
        num_hashes: int = 260,
        seed: int = 42,
        use_64bit_hash: bool = False,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        sample_id_field: str = "sample_id",
        position_field: str = "position",
        modality_field: str = "modality",
        text_content_field: str = "text_content",
        text_modality: str = "text",
        metadata_modality: str = "metadata",
        metadata_json_path: str | None = "$.content",
        text_separator: str = "\n\n",
        pool: bool = True,
    ):
        if text_mode not in ("metadata_content", "text_rows"):
            msg = "text_mode must be one of {'metadata_content', 'text_rows'}"
            raise ValueError(msg)

        super().__init__(
            output_path=output_path,
            text_field=text_field,
            minhash_field=minhash_field,
            char_ngrams=char_ngrams,
            num_hashes=num_hashes,
            seed=seed,
            use_64bit_hash=use_64bit_hash,
            read_format="parquet",
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
            pool=pool,
        )
        self.text_mode = text_mode
        self.sample_id_field = sample_id_field
        self.position_field = position_field
        self.modality_field = modality_field
        self.text_content_field = text_content_field
        self.text_modality = text_modality
        self.metadata_modality = metadata_modality
        self.metadata_json_path = metadata_json_path
        self.text_separator = text_separator

    def _read_interleaved(self, filepaths: list[str]) -> cudf.DataFrame:
        read_kwargs = self.read_kwargs.copy()
        if read_kwargs.get("columns") is not None:
            msg = "Columns cannot be set in read_kwargs for InterleavedMinHashStage"
            raise ValueError(msg)

        columns = [self.sample_id_field, self.modality_field, self.text_content_field]
        if self.text_mode == "text_rows":
            columns.append(self.position_field)
        return self.read_parquet(filepath=filepaths, columns=columns, assign_id=False, **read_kwargs)

    def _extract_metadata_content(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df[df[self.modality_field] == self.metadata_modality][[self.sample_id_field, self.text_content_field]]
        row_count_field = "_metadata_row_count"
        df[row_count_field] = 1
        row_counts = df.groupby(self.sample_id_field, sort=True).agg({row_count_field: "sum"}).reset_index()
        duplicate_metadata_samples = row_counts[row_counts[row_count_field] > 1]
        if len(duplicate_metadata_samples) > 0:
            duplicate_sample_ids = duplicate_metadata_samples[self.sample_id_field].head().to_arrow().to_pylist()
            msg = (
                "Found samples with more than one metadata row while using text_mode='metadata_content'. "
                f"Example sample_ids: {duplicate_sample_ids}"
            )
            raise ValueError(msg)

        if len(df) == 0:
            return cudf.DataFrame(
                {
                    self.sample_id_field: cudf.Series([], dtype=df[self.sample_id_field].dtype),
                    self.text_field: cudf.Series([], dtype="str"),
                }
            )
        if self.metadata_json_path is None:
            df[self.text_field] = df[self.text_content_field]
        else:
            df[self.text_field] = df[self.text_content_field].str.get_json_object(self.metadata_json_path)
        return df[[self.sample_id_field, self.text_field]]

    def _extract_text_rows(self, df: cudf.DataFrame) -> cudf.DataFrame:
        df = df[df[self.modality_field] == self.text_modality][
            [self.sample_id_field, self.position_field, self.text_content_field]
        ]
        df = df[df[self.text_content_field].notnull()]
        if len(df) == 0:
            return cudf.DataFrame(
                {
                    self.sample_id_field: cudf.Series([], dtype=df[self.sample_id_field].dtype),
                    self.text_field: cudf.Series([], dtype="str"),
                }
            )

        df = df.sort_values([self.sample_id_field, self.position_field])
        df = df.groupby(self.sample_id_field, sort=True).agg({self.text_content_field: list})
        df[self.text_field] = df[self.text_content_field].str.join(self.text_separator)
        df = df.reset_index()
        return df[[self.sample_id_field, self.text_field]]

    def sample_ordering(self, df: cudf.DataFrame) -> cudf.Series:
        """Return sorted unique sample_ids in the exact order used for _curator_id assignment.

        This is the contract between InterleavedMinHashStage and InterleavedDuplicatesRemovalStage:
        both call this method on their (independently read) partition and must produce the
        identical Series, so the partition's _curator_id range can be reconstructed by positional
        lookup at removal time.
        """
        return sample_ordering(df, self.sample_id_field)

    def _extract_documents_with_metrics(self, df: cudf.DataFrame) -> tuple[cudf.DataFrame, dict[str, float]]:
        metrics = {}

        sample_ordering_t0 = time.perf_counter()
        documents = self.sample_ordering(df).to_frame(name=self.sample_id_field)
        metrics["sample_ordering_time"] = time.perf_counter() - sample_ordering_t0
        if len(documents) == 0:
            msg = "No interleaved samples found"
            raise ValueError(msg)

        text_extraction_t0 = time.perf_counter()
        if self.text_mode == "metadata_content":
            extracted_text = self._extract_metadata_content(df)
        elif self.text_mode == "text_rows":
            extracted_text = self._extract_text_rows(df)
        else:
            msg = f"Unsupported text_mode: {self.text_mode}"
            raise ValueError(msg)
        text_extraction_time = time.perf_counter() - text_extraction_t0
        metrics["text_extraction_time"] = text_extraction_time
        metrics[f"{self.text_mode}_extraction_time"] = text_extraction_time
        metrics["num_extracted_text_documents"] = float(len(extracted_text))

        sample_text_join_t0 = time.perf_counter()
        documents = documents.merge(extracted_text, on=self.sample_id_field, how="left")
        documents = documents.sort_values(self.sample_id_field).reset_index(drop=True)
        metrics["sample_text_join_time"] = time.perf_counter() - sample_text_join_t0
        return documents, metrics

    def _extract_documents(self, df: cudf.DataFrame) -> cudf.DataFrame:
        documents, metrics = self._extract_documents_with_metrics(df)
        self._log_metrics(metrics)
        return documents

    def process(self, task: FileGroupTask) -> FileGroupTask:
        """
        Process interleaved files to compute sample-level minhashes.

        Args:
            task: FileGroupTask containing interleaved Parquet file paths

        Returns:
            FileGroupTask containing paths to sample-level minhash output files
        """

        if self.minhash_processor is None or self.id_generator is None:
            msg = "MinHash processor or ID generator not initialized. Call setup() first."
            raise RuntimeError(msg)

        output_file = self.output_fs.sep.join([self.output_path, f"{task._uuid}.parquet"])

        metrics = {"num_input_files": float(len(task.data))}

        read_t0 = time.perf_counter()
        df = self._read_interleaved(task.data)
        metrics["read_time"] = time.perf_counter() - read_t0
        metrics["num_input_rows"] = float(len(df))

        normalization_t0 = time.perf_counter()
        documents, normalization_metrics = self._extract_documents_with_metrics(df)
        metrics.update(normalization_metrics)
        metrics["normalization_time"] = time.perf_counter() - normalization_t0
        metrics["num_documents"] = float(len(documents))

        assign_id_t0 = time.perf_counter()
        documents = self.assign_id(task.data, documents)
        metrics["assign_id_time"] = time.perf_counter() - assign_id_t0

        hashable_filter_t0 = time.perf_counter()
        hashable_documents = documents[documents[self.text_field].notnull()]
        metrics["hashable_filter_time"] = time.perf_counter() - hashable_filter_t0
        metrics["num_hashable_documents"] = float(len(hashable_documents))
        metrics["num_skipped_documents"] = metrics["num_documents"] - metrics["num_hashable_documents"]

        result_df = hashable_documents[[CURATOR_DEDUP_ID_STR]]
        minhash_t0 = time.perf_counter()
        result_df[self.minhash_field] = self.minhash_processor.compute_minhashes(hashable_documents[self.text_field])
        metrics["minhash_time"] = time.perf_counter() - minhash_t0

        write_t0 = time.perf_counter()
        self.write_parquet(df=result_df, filepath=output_file, **self.write_kwargs)
        metrics["write_time"] = time.perf_counter() - write_t0

        self._log_metrics(metrics)
        logger.debug(
            "InterleavedMinHashStage task={} rows={} samples={} hashable={} timings={}",
            task.task_id,
            int(metrics["num_input_rows"]),
            int(metrics["num_documents"]),
            int(metrics["num_hashable_documents"]),
            {
                "read_time": metrics["read_time"],
                "normalization_time": metrics["normalization_time"],
                "sample_text_join_time": metrics["sample_text_join_time"],
                "assign_id_time": metrics["assign_id_time"],
                "minhash_time": metrics["minhash_time"],
                "write_time": metrics["write_time"],
            },
        )

        return FileGroupTask(
            task_id=f"{task.task_id}",
            dataset_name=f"{task.dataset_name}_interleaved_minhash",
            data=[output_file],
            _metadata={
                **task._metadata,
                "minhash_field": self.minhash_field,
                "num_hashes": self.num_hashes,
                "storage_options": self.write_kwargs.get("storage_options"),
                "text_mode": self.text_mode,
                "num_documents": len(documents),
                "num_hashable_documents": len(hashable_documents),
                "interleaved_minhash_metrics": metrics,
            },
            _stage_perf=task._stage_perf,
        )
