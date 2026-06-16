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

from typing import Any

import pyarrow as pa


def sample_ordering(data: Any, sample_id_field: str = "sample_id") -> Any:  # noqa: ANN401
    """Return sample IDs in the canonical order used for interleaved dedup IDs."""
    if isinstance(data, pa.Table):
        return sorted(set(data.column(sample_id_field).to_pylist()))
    return data[sample_id_field].drop_duplicates().sort_values().reset_index(drop=True)
