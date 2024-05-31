# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import time

import dask_cudf

from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.script_utils import parse_gpu_dedup_args


def attach_args(parser=None):
    description = """Takes the buckets generated from minhashes and uses
    document length information to create a coarse mapping of mapping multiple
    buckets to a logical partition by using a modified bin packing algorithm.
    """
    if not parser:
        parser = parse_gpu_dedup_args(description=description)
    parser.add_argument(
        "--input-bucket-mapping-dir",
        type=str,
        help="The directory containing anchor docs with bk files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output dir to write results",
    )
    return parser


def map_buckets_to_cc(
    input_bucket_path,
    output_jaccard_similarity_path,
):
    """
    Workflow for jaccard shuffle
    Args:
        client: dask client
        input_bucket_path: path to input buckets
    """

    anchor_docs_ddf = dask_cudf.read_parquet(input_bucket_path, split_row_groups=False)

    int_id_to_str_cols = {
        ("dataset_id", "doc_id"): "adlr_id",
        ("anchor_1_dataset_id", "anchor_1_doc_id"): "anchor_1_adlr_id",
        ("anchor_0_dataset_id", "anchor_0_doc_id"): "anchor_0_adlr_id",
    }

    for source_cols, target_col in int_id_to_str_cols.items():
        c0, c1 = source_cols
        anchor_docs_ddf[target_col] = (
            anchor_docs_ddf[c0].astype(str) + "-" + anchor_docs_ddf[c1].astype(str)
        )
        anchor_docs_ddf = anchor_docs_ddf.drop(columns=[c0, c1])

    x = anchor_docs_ddf[["adlr_id", "anchor_0_adlr_id"]]
    y = anchor_docs_ddf[["adlr_id", "anchor_1_adlr_id"]]
    x.columns = ["adlr_id_x", "adlr_id_y"]
    y.columns = ["adlr_id_x", "adlr_id_y"]
    res_df = dask_cudf.concat([x, y])
    res_df["jaccard"] = 1

    res_df.to_parquet(
        output_jaccard_similarity_path,
        write_index=False,
    )


def main(args):
    input_bucket_path = args.input_bucket_mapping_dir
    OUTPUT_PATH = args.output_dir
    output_jaccard_similarity_path = os.path.join(
        OUTPUT_PATH, "jaccard_similarity_results.parquet"
    )
    client = get_client(args, "gpu")
    print(f"Num Workers = {get_num_workers(client)}", flush=True)
    print("Connected to dask cluster", flush=True)
    print(
        "Running buckets -> CC compatible jaccard similarity (no fp check)", flush=True
    )
    print(f"Args = {args}")
    st = time.time()
    map_buckets_to_cc(
        input_bucket_path,
        output_jaccard_similarity_path,
    )
    et = time.time()
    print(f"Bucket Mapping time taken = {et-st} s")


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
