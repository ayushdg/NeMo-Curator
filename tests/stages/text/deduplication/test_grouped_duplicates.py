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

from nemo_curator.tasks import FileGroupTask

with suppress(ImportError):
    import cudf

    from nemo_curator.stages.deduplication.fuzzy.utils import CURATOR_FUZZY_DUPLICATE_GROUP_FIELD
    from nemo_curator.stages.deduplication.fuzzy.workflow import (
        ID_GENERATOR_OUTPUT_FILENAME,
        FuzzyDeduplicationWorkflow,
    )
    from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
    from nemo_curator.stages.text.deduplication.grouped_duplicates_workflow import WriteDuplicateGroupsWorkflow


@pytest.mark.gpu
@pytest.mark.usefixtures("shared_ray_client")
class TestWriteDuplicateGroupsWorkflow:
    """End-to-end: fuzzy identify, then the disk-based (sort→materialize→merge) grouped writer."""

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
        tasks = [FileGroupTask(task_id="fg0", dataset_name="test", data=files, _metadata={"source_files": files})]

        cache_path = tmp_path / "cache"
        output_path = tmp_path / "output"
        cache_path.mkdir(exist_ok=True)

        FuzzyDeduplicationWorkflow(
            cache_path=str(cache_path),
            output_path=str(output_path),
            input_filetype="parquet",
            text_field="text",
            seed=42,
            char_ngrams=5,
            num_bands=5,
            minhashes_per_band=1,
            bands_per_iteration=5,
        ).run(initial_tasks=tasks)

        grouped_out = tmp_path / "grouped"
        result = WriteDuplicateGroupsWorkflow(
            output_path=str(grouped_out),
            id_generator_path=str(output_path / ID_GENERATOR_OUTPUT_FILENAME),
            connected_components_path=str(cache_path / "ConnectedComponentsStage"),
            input_filetype="parquet",
            n_map=2,
            num_output_partitions=4,
        ).run(initial_tasks=tasks)

        assert result.get_metadata("num_duplicate_docs_written") == 5

        out_dir = grouped_out / "DuplicateGroups"
        out_df = cudf.read_parquet(out_dir).to_pandas()
        assert {"id", "text", CURATOR_DEDUP_ID_STR, CURATOR_FUZZY_DUPLICATE_GROUP_FIELD}.issubset(out_df.columns)

        grouped = [set(g) for g in out_df.groupby(CURATOR_FUZZY_DUPLICATE_GROUP_FIELD)["id"].agg(list).tolist()]
        expected = [{4, -1}, {1, 2, 300}]
        assert sorted([sorted(g) for g in grouped]) == sorted([sorted(e) for e in expected])

        # Each duplicate group must land entirely within a single output file.
        group_to_files: dict[int, set[str]] = {}
        for pf in out_dir.glob("*.parquet"):
            gdf = pd.read_parquet(pf)
            for gid in gdf[CURATOR_FUZZY_DUPLICATE_GROUP_FIELD].unique():
                group_to_files.setdefault(int(gid), set()).add(pf.name)
        assert all(len(file_set) == 1 for file_set in group_to_files.values())

        # Intermediate artifacts exist (sorted CC + runs).
        assert (grouped_out / "ConnectedComponentsSortedById").exists()
        assert (grouped_out / "DuplicateRuns").exists()
