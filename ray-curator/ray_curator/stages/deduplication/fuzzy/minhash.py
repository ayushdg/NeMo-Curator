import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import cudf
import numpy as np
import ray
import rmm
from ray.util.actor_pool import ActorPool

from ray_curator.utils.file_utils import get_all_files_paths_under
from ray_curator.utils.id_generator import CURATOR_ID_STR, IdGenerator
from ray_curator.utils.io_utils import RayIO
from ray_curator.utils.ray_utils import get_num_gpus


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


# Create a Ray Actor
@ray.remote(num_gpus=1)
class GPUMinhashActor(MinHash, RayIO):
    def __init__(  # noqa: PLR0913
        self,
        seed: int = 42,
        num_hashes: int = 260,
        char_ngrams: int = 24,
        use_64bit_hash: bool = False,
        pool: bool = True,
        id_generator: "IdGenerator | None" = None,
    ):
        # Initialize parent class
        MinHash.__init__(
            self,
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
        )
        RayIO.__init__(
            self,
            id_generator=id_generator,
        )

        # Initialize memory pool for cuDF
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

    def __call__(  # noqa: PLR0913
        self,
        infiles: str | list[str],
        outfile: str,
        text_column: str,
        read_func: Callable | None = None,
        columns: list[str] | None = None,
        minhash_column: str = "_minhash_signature",
    ):
        """
        Process an input file, compute minhashes, and write results to an output file.
        Automatically adds a unique _curator_id field to each document if not present.

        Parameters
        ----------
        infiles: str, list[str]
            Path or list of paths to input file(s)
        outfile: str
            Path to output file (Parquet format)
        read_func: Callable, optional
            Function to read input file(s)
        columns: list, optional
            Columns to read from input file
        text_column: str, optional
            Column containing text to hash. If None, uses self.text_field
        """
        if columns is None:
            columns = [text_column]

        # Read input file
        if read_func is None:
            df = self.read_jsonl(infiles, columns=columns, assign_id=True)
        else:
            df = self.custom_read(filepath=infiles, read_func=read_func, assign_id=True)

        result_df = df[[CURATOR_ID_STR]]

        # Compute minhashes on the text column and add to dataframe
        result_df[minhash_column] = self.compute_minhashes(df[text_column])

        # Write output file
        self.write_parquet(result_df, outfile)


def minhash(  # noqa: PLR0913
    input_data_dir: str,
    output_minhash_dir: str,
    text_column: str,
    files_per_partition: int,
    read_func: Callable,
    char_ngrams: int = 24,
    use_64bit_hash: bool = False,
    seed: int = 42,
    num_hashes: int = 260,
) -> None:
    num_gpus = get_num_gpus()
    curator_id_generator = IdGenerator.options(name="fuzzy_dedup_id_generator").remote()

    input_files = get_all_files_paths_under(input_data_dir)
    input_batches = [input_files[i : i + files_per_partition] for i in range(0, len(input_files), files_per_partition)]
    print(f"Processing a total of {len(input_files)} files in {len(input_batches)} batches")
    output_files = [os.path.join(output_minhash_dir, f"partition_{i}.parquet") for i in range(len(input_batches))]

    os.makedirs(os.path.dirname(output_files[0]), exist_ok=True)

    start = time.time()
    # Pass the shared ID generator to all actors
    minhash_actors = [
        GPUMinhashActor.remote(
            id_generator=curator_id_generator,  # Still pass it, but it will only be used if needed
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
            num_hashes=num_hashes,
            seed=seed,
        )
        for _ in range(num_gpus)
    ]

    minhash_actor_pool = ActorPool(minhash_actors)

    results = minhash_actor_pool.map_unordered(
        lambda actor, values: actor.__call__.remote(
            infiles=values[0],
            read_func=read_func,
            outfile=values[1],
            columns=[text_column],
            text_column=text_column,
        ),
        zip(input_batches, output_files, strict=False),
    )
    results = list(results)
    print(len(results))
    end = time.time()
    print(f"Time taken to compute minhashes: {end - start} seconds")
    del minhash_actor_pool
    for actor in minhash_actors:
        ray.kill(actor)
