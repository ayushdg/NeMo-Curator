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

from contextlib import suppress
from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.stages.deduplication.fuzzy.reverse_id_index import ReverseIdIndex
from nemo_curator.tasks import FileGroupTask

# Suppress GPU-related import errors when running pytest -m "not gpu"
with suppress(ImportError):
    import cudf

    from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
    from nemo_curator.stages.deduplication.fuzzy.workflow import (
        ID_GENERATOR_OUTPUT_FILENAME,
        FuzzyDeduplicationWorkflow,
    )
    from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
    from nemo_curator.stages.text.deduplication.write_duplicates import WriteTextDuplicatesStage
    from nemo_curator.stages.text.deduplication.write_duplicates_workflow import WriteTextDuplicatesWorkflow


class TestReverseIdIndex:
    """CPU-only tests for the driver-side reverse index."""

    def test_single_group_lookup(self) -> None:
        index = ReverseIdIndex([(0, 4, ["a.parquet", "b.parquet"])])
        assert len(index) == 1
        assert index.group_indices_for_ids([0, 3]) == [0]
        assert index.files(0) == ["a.parquet", "b.parquet"]
        assert index.min_id(0) == 0
        assert index.max_id(0) == 4

    def test_multi_group_lookup_and_dedup(self) -> None:
        # Deliberately pass ranges out of order to verify sorting by min_id.
        index = ReverseIdIndex([(10, 19, ["c"]), (0, 4, ["a"]), (5, 9, ["b"])])
        # ids 1 -> group [0,4], 7 -> [5,9], 15 -> [10,19]; 7 appears twice (dedup).
        assert index.group_indices_for_ids([1, 7, 7, 15]) == [0, 1, 2]
        assert index.files(0) == ["a"]
        assert index.files(1) == ["b"]
        assert index.files(2) == ["c"]

    def test_ids_outside_ranges_are_ignored(self) -> None:
        index = ReverseIdIndex([(0, 4, ["a"]), (10, 14, ["b"])])
        # 5-9 fall in the gap, 20 is past the end, -1 is before the start.
        assert index.group_indices_for_ids([-1, 5, 9, 20]) == []
        assert index.group_indices_for_ids([2, 12]) == [0, 1]

    def test_empty_inputs(self) -> None:
        assert ReverseIdIndex([]).group_indices_for_ids([1, 2]) == []
        assert ReverseIdIndex([(0, 4, ["a"])]).group_indices_for_ids([]) == []


@pytest.mark.gpu
class TestWriteTextDuplicatesStage:
    """GPU tests that drive WriteTextDuplicatesStage.process directly (no executor / id generator)."""

    @staticmethod
    def _write_dup_partition(path: Path, ids: list[int], groups: list[int]) -> FileGroupTask:
        pd.DataFrame(
            {CURATOR_DEDUP_ID_STR: ids, CURATOR_FUZZY_DUPLICATE_GROUP_FIELD: groups}
        ).to_parquet(path, index=False)
        return FileGroupTask(task_id="0", dataset_name="t", data=[str(path)])

    def test_full_groups_single_input_group(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        f1 = input_dir / "part1.parquet"
        f2 = input_dir / "part2.parquet"
        pd.DataFrame({"doc_id": ["a", "b", "c"], "text": ["t0", "t1", "t2"]}).to_parquet(f1, index=False)
        pd.DataFrame({"doc_id": ["d", "e"], "text": ["t3", "t4"]}).to_parquet(f2, index=False)
        # One group of 2 files, 5 rows -> ids 0..4 positionally.
        reverse_index = ReverseIdIndex([(0, 4, [str(f1), str(f2)])])

        task = self._write_dup_partition(tmp_path / "part.0.parquet", [0, 1, 2, 3, 4], [10, 10, 10, 20, 20])
        stage = WriteTextDuplicatesStage(reverse_index=reverse_index, output_path=str(tmp_path / "out"))
        out_task = stage.process(task)

        out_df = cudf.read_parquet(out_task.data).to_pandas()
        assert len(out_df) == 5
        assert set(out_df.columns) == {"doc_id", "text", CURATOR_DEDUP_ID_STR, CURATOR_FUZZY_DUPLICATE_GROUP_FIELD}
        groups = out_df.groupby(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)["doc_id"].agg(set).to_dict()
        assert groups == {10: {"a", "b", "c"}, 20: {"d", "e"}}
        # Output is sorted by group then id.
        assert out_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].tolist() == sorted(
            out_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].tolist()
        )
        assert out_task._metadata["num_duplicates_written"] == 5
        assert out_task._metadata["num_input_groups_read"] == 1

    def test_subset_and_cross_file_group(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        fa = input_dir / "a.parquet"
        fb = input_dir / "b.parquet"
        pd.DataFrame({"doc_id": ["a0", "a1", "a2"]}).to_parquet(fa, index=False)
        pd.DataFrame({"doc_id": ["b0", "b1", "b2"]}).to_parquet(fb, index=False)
        # Two separate input groups: [0,2] and [3,5].
        reverse_index = ReverseIdIndex([(0, 2, [str(fa)]), (3, 5, [str(fb)])])

        # Duplicates are a subset spanning both groups; ids 0, 2, 3 are NOT duplicates.
        task = self._write_dup_partition(tmp_path / "part.0.parquet", [1, 4, 5], [10, 10, 10])
        stage = WriteTextDuplicatesStage(reverse_index=reverse_index, output_path=str(tmp_path / "out"))
        out_task = stage.process(task)

        out_df = cudf.read_parquet(out_task.data).to_pandas()
        assert set(out_df[CURATOR_DEDUP_ID_STR].tolist()) == {1, 4, 5}
        assert set(out_df["doc_id"].tolist()) == {"a1", "b1", "b2"}
        assert (out_df[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD] == 10).all()
        assert out_task._metadata["num_input_groups_read"] == 2

    def test_id_count_mismatch_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "a.parquet"
        pd.DataFrame({"doc_id": ["a", "b", "c"]}).to_parquet(f, index=False)
        # Claim 10 ids for a 3-row file -> positional reproduction would be wrong.
        reverse_index = ReverseIdIndex([(0, 9, [str(f)])])
        task = self._write_dup_partition(tmp_path / "part.0.parquet", [0], [10])
        stage = WriteTextDuplicatesStage(reverse_index=reverse_index, output_path=str(tmp_path / "out"))
        with pytest.raises(RuntimeError, match="diverged from the minhash read"):
            stage.process(task)


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestWriteTextDuplicatesWorkflow:
    """End-to-end: run fuzzy dedup, then write the duplicate documents grouped by group id."""

    def test_e2e(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "id": [1, 2, 300, 4, -1],
                "text": [
                    "A test string",
                    "A different test string",
                    "A different object",
                    "The quick brown fox jumps over the lazy dog",
                    "The quick black cat jumps over the lazy dog",
                ],
            }
        )
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        f1 = input_dir / "part1.parquet"
        f2 = input_dir / "part2.parquet"
        df.iloc[:3].to_parquet(f1, index=False)
        df.iloc[3:].to_parquet(f2, index=False)
        files = [str(f1), str(f2)]
        tasks = [
            FileGroupTask(task_id="fg0", dataset_name="test", data=files, _metadata={"source_files": files})
        ]

        cache_path = tmp_path / "cache"
        output_path = tmp_path / "output"
        cache_path.mkdir(exist_ok=True)

        FuzzyDeduplicationWorkflow(
            cache_path=str(cache_path),
            output_path=str(output_path),
            input_filetype="parquet",
            text_field="text",
            perform_removal=False,
            seed=42,
            char_ngrams=5,
            num_bands=5,
            minhashes_per_band=1,
            bands_per_iteration=5,
        ).run(initial_tasks=tasks)

        dupes_out = tmp_path / "dupes"
        result = WriteTextDuplicatesWorkflow(
            output_path=str(dupes_out),
            id_generator_path=str(output_path / ID_GENERATOR_OUTPUT_FILENAME),
            connected_components_path=str(cache_path / "ConnectedComponentsStage"),
            input_filetype="parquet",
        ).run(initial_tasks=tasks)

        # Every document here is part of a duplicate group, so all 5 are written.
        assert result.get_metadata("num_duplicates_written") == 5

        out_dir = dupes_out / "WriteTextDuplicatesStage"
        out_df = cudf.read_parquet(out_dir).to_pandas()
        assert {"id", "text", CURATOR_DEDUP_ID_STR, CURATOR_FUZZY_DUPLICATE_GROUP_FIELD}.issubset(out_df.columns)

        grouped = [
            set(g) for g in out_df.groupby(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)["id"].agg(list).tolist()
        ]
        expected = [{4, -1}, {1, 2, 300}]
        assert sorted([sorted(g) for g in grouped]) == sorted([sorted(e) for e in expected])

        # Each duplicate group must land entirely within a single output file.
        group_to_files: dict[int, set[str]] = {}
        for pf in out_dir.glob("*.parquet"):
            gdf = pd.read_parquet(pf)
            for group_id in gdf[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].unique():
                group_to_files.setdefault(int(group_id), set()).add(pf.name)
        assert all(len(file_set) == 1 for file_set in group_to_files.values())
