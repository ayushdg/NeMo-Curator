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

"""Driver-side reverse index mapping a ``_curator_dedup_id`` back to its input file group.

Kept free of GPU dependencies so it can be built and tested on the driver/CPU. Shared by the text
(and, later, interleaved) duplicate-writing stages.
"""

import numpy as np


class ReverseIdIndex:
    """Maps a ``_curator_dedup_id`` back to the input file group that produced it.

    Built on the driver from the exact ``FileGroupTask``\\ s that minhash/identification ran on,
    paired with the contiguous id range the id generator assigned to each group. During minhash a
    group's documents are assigned ``arange(min_id, max_id + 1)`` in file order, so a group can be
    reproduced by reading its files (in order) and re-assigning that range positionally.

    The ranges are sorted by ``min_id`` and assumed disjoint (the id generator allocates a fresh
    contiguous block per registered batch), enabling a vectorized ``searchsorted`` lookup.
    """

    def __init__(self, ranges: list[tuple[int, int, list[str]]]):
        sorted_ranges = sorted(ranges, key=lambda r: r[0])
        self._min_ids = np.asarray([r[0] for r in sorted_ranges], dtype="int64")
        self._max_ids = np.asarray([r[1] for r in sorted_ranges], dtype="int64")
        self._files = [list(r[2]) for r in sorted_ranges]

    def __len__(self) -> int:
        return len(self._min_ids)

    def files(self, idx: int) -> list[str]:
        return self._files[idx]

    def min_id(self, idx: int) -> int:
        return int(self._min_ids[idx])

    def max_id(self, idx: int) -> int:
        return int(self._max_ids[idx])

    def group_indices_for_ids(self, ids: np.ndarray) -> list[int]:
        """Return the (sorted, deduplicated) indices of the groups that contain any of ``ids``."""
        ids = np.asarray(ids, dtype="int64")
        if len(self._min_ids) == 0 or len(ids) == 0:
            return []
        # For each id find the rightmost group whose min_id <= id, then verify id <= that max_id.
        pos = np.searchsorted(self._min_ids, ids, side="right") - 1
        valid = pos >= 0
        clipped = np.clip(pos, 0, len(self._max_ids) - 1)
        valid &= ids <= self._max_ids[clipped]
        return np.unique(pos[valid]).tolist()
