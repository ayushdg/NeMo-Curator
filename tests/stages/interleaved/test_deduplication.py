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

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.stages.interleaved.deduplication.removal import InterleavedSampleDuplicatesRemovalStage
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch


def _interleaved_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sample_id": "s_b", "position": -1, "modality": "metadata", "text_content": None},
            {"sample_id": "s_b", "position": 0, "modality": "text", "text_content": "duplicate text"},
            {"sample_id": "s_a", "position": -1, "modality": "metadata", "text_content": None},
            {"sample_id": "s_a", "position": 0, "modality": "text", "text_content": "keep text"},
            {"sample_id": "s_b", "position": 1, "modality": "image", "text_content": None},
        ]
    )


@pytest.mark.usefixtures("ray_client_with_id_generator")
def test_interleaved_parquet_reader_generates_and_assigns_sample_ids(tmp_path: Path) -> None:
    input_file = tmp_path / "interleaved.parquet"
    _interleaved_rows().to_parquet(input_file, index=False)
    task = FileGroupTask(task_id="input", dataset_name="test", data=[str(input_file)])

    generate_stage = InterleavedParquetReaderStage(_generate_ids=True)
    generate_stage.setup()
    generated = generate_stage.process(task)
    assert isinstance(generated, InterleavedBatch)
    generated_df = generated.to_pandas()

    sample_to_id = (
        generated_df[["sample_id", CURATOR_DEDUP_ID_STR]]
        .drop_duplicates()
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    assert sample_to_id["sample_id"].tolist() == ["s_a", "s_b"]
    assert sample_to_id[CURATOR_DEDUP_ID_STR].tolist() == [0, 1]
    s_b_ids = generated_df.loc[generated_df["sample_id"] == "s_b", CURATOR_DEDUP_ID_STR].tolist()
    assert len(set(s_b_ids)) == 1

    assign_stage = InterleavedParquetReaderStage(_assign_ids=True)
    assign_stage.setup()
    assigned = assign_stage.process(task)
    assert isinstance(assigned, InterleavedBatch)
    assigned_df = assigned.to_pandas()

    pd.testing.assert_series_equal(
        generated_df[CURATOR_DEDUP_ID_STR],
        assigned_df[CURATOR_DEDUP_ID_STR],
        check_names=False,
    )


def test_interleaved_sample_duplicates_removal_drops_all_sample_rows(tmp_path: Path) -> None:
    df = _interleaved_rows()
    df[CURATOR_DEDUP_ID_STR] = df["sample_id"].map({"s_a": 0, "s_b": 1})
    task = InterleavedBatch(task_id="input", dataset_name="test", data=df)

    duplicate_dir = tmp_path / "duplicates"
    duplicate_dir.mkdir()
    pd.DataFrame({CURATOR_DEDUP_ID_STR: [1]}).to_parquet(duplicate_dir / "part.0.parquet", index=False)

    stage = InterleavedSampleDuplicatesRemovalStage(ids_to_remove_path=str(duplicate_dir))
    result = stage.process(task)
    result_df = result.to_pandas()

    assert set(result_df["sample_id"].tolist()) == {"s_a"}
    assert CURATOR_DEDUP_ID_STR not in result_df.columns
    assert result._metadata["num_samples_removed"] == 1
    assert result._metadata["num_rows_in"] == 5
    assert result._metadata["num_rows_out"] == 2


def test_interleaved_sample_duplicates_removal_process_batch_accepts_arrow_columns(tmp_path: Path) -> None:
    df = _interleaved_rows()
    df[CURATOR_DEDUP_ID_STR] = df["sample_id"].map({"s_a": 0, "s_b": 1})
    task = InterleavedBatch(task_id="input", dataset_name="test", data=pa.Table.from_pandas(df, preserve_index=False))

    duplicate_dir = tmp_path / "duplicates"
    duplicate_dir.mkdir()
    pd.DataFrame({CURATOR_DEDUP_ID_STR: [1]}).to_parquet(duplicate_dir / "part.0.parquet", index=False)

    stage = InterleavedSampleDuplicatesRemovalStage(ids_to_remove_path=str(duplicate_dir))
    results = stage.process_batch([task])
    assert len(results) == 1

    result_df = results[0].to_pandas()
    assert set(result_df["sample_id"].tolist()) == {"s_a"}
    assert CURATOR_DEDUP_ID_STR not in result_df.columns
    assert results[0]._metadata["num_samples_removed"] == 1


def test_interleaved_sample_duplicates_removal_missing_duplicate_path_keeps_input(tmp_path: Path) -> None:
    df = _interleaved_rows()
    df[CURATOR_DEDUP_ID_STR] = df["sample_id"].map({"s_a": 0, "s_b": 1})
    task = InterleavedBatch(task_id="input", dataset_name="test", data=df)

    stage = InterleavedSampleDuplicatesRemovalStage(ids_to_remove_path=str(tmp_path / "missing"))
    result = stage.process(task)
    result_df = result.to_pandas()

    assert set(result_df["sample_id"].tolist()) == {"s_a", "s_b"}
    assert CURATOR_DEDUP_ID_STR not in result_df.columns
    assert result._metadata["num_samples_removed"] == 0
    assert result._metadata["num_rows_out"] == len(df)
