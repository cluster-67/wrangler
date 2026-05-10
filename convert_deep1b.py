#!/usr/bin/env python3
import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm


HEADER_DTYPE = np.dtype([("count", "<u4"), ("dim", "<u4")])


def read_header(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=HEADER_DTYPE, count=1)
    if header.size != 1:
        raise ValueError(f"Failed to read header from {path}")
    return int(header[0]["count"]), int(header[0]["dim"])


def memmap_fbin(path, dtype):
    count, dim = read_header(path)
    offset = HEADER_DTYPE.itemsize
    expected_bytes = offset + (count * dim * np.dtype(dtype).itemsize)
    actual_bytes = os.path.getsize(path)
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"File too small for declared header: {path}. "
            f"Expected {expected_bytes} bytes, found {actual_bytes}."
        )
    mmap = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count, dim))
    return mmap, count, dim


def read_fbin(path):
    count, dim = read_header(path)
    with open(path, "rb") as f:
        f.seek(HEADER_DTYPE.itemsize)
        data = np.fromfile(f, dtype=np.float32, count=count * dim)
    if data.size != count * dim:
        raise ValueError(f"Unexpected data size for {path}")
    return data.reshape(count, dim)


def read_ibin(path):
    count, dim = read_header(path)
    with open(path, "rb") as f:
        f.seek(HEADER_DTYPE.itemsize)
        data = np.fromfile(f, dtype=np.int32, count=count * dim)
    if data.size != count * dim:
        raise ValueError(f"Unexpected data size for {path}")
    return data.reshape(count, dim)


def chunked_write_train(h5f, base_mmap, chunk_rows):
    total_rows, dim = base_mmap.shape
    train_ds = h5f.create_dataset(
        "train",
        shape=(total_rows, dim),
        dtype=np.float32,
        chunks=(chunk_rows, dim),
    )
    for start in tqdm(range(0, total_rows, chunk_rows), desc="Writing /train"):
        end = min(start + chunk_rows, total_rows)
        train_ds[start:end] = base_mmap[start:end]
    return train_ds


def write_neighbors_and_distances(h5f, query, base_mmap, neighbors):
    num_queries = query.shape[0]
    top_k = neighbors.shape[1]
    neighbors_ds = h5f.create_dataset(
        "neighbors",
        shape=(num_queries, top_k),
        dtype=np.int32,
    )
    distances_ds = h5f.create_dataset(
        "distances",
        shape=(num_queries, top_k),
        dtype=np.float32,
    )

    oob_count = 0
    oob_queries = 0
    for i in tqdm(range(num_queries), desc="Writing /neighbors and /distances"):
        neighbor_row = neighbors[i].astype(np.int32, copy=False)
        valid_mask = neighbor_row < base_mmap.shape[0]
        valid_neighbors = neighbor_row[valid_mask]

        neighbors_out = np.full(top_k, -1, dtype=np.int32)
        distances_out = np.full(top_k, np.inf, dtype=np.float32)

        if valid_neighbors.size:
            neighbor_vectors = base_mmap[valid_neighbors]
            diff = neighbor_vectors - query[i]
            distances = np.einsum("ij,ij->i", diff, diff, dtype=np.float32)
            neighbors_out[valid_mask] = valid_neighbors
            distances_out[valid_mask] = distances.astype(np.float32, copy=False)

        oob_in_row = int((~valid_mask).sum())
        if oob_in_row:
            oob_queries += 1
            oob_count += oob_in_row
        neighbors_ds[i] = neighbors_out
        distances_ds[i] = distances_out

    if oob_count:
        print(
            f"ERROR: {oob_queries} queries had {oob_count} neighbors out of bounds (set to -1).",
            file=sys.stderr,
        )


def validate_input_paths(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        missing_str = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_str}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Deep1B fbin/ibin to HDF5")
    parser.add_argument("--size", required=True, choices=["10M", "1B"], help="Dataset size")
    parser.add_argument(
        "--input-dir",
        default=os.path.join(os.environ.get("SCRATCH", ""), "data", "deep1b"),
        help="Directory containing Deep1B files",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HDF5 path (default: $SCRATCH/data/deep1b/deep1b-{size}.hdf5)",
    )
    parser.add_argument(
        "--train-chunk-rows",
        type=int,
        default=10000,
        help="Rows per chunk for /train dataset",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input_dir:
        raise ValueError("--input-dir must be set or SCRATCH must be defined")

    base_file = f"base.{args.size}.fbin"
    query_file = "query.public.10K.fbin"
    gt_file = "groundtruth.public.10K.ibin"

    base_path = os.path.join(args.input_dir, base_file)
    query_path = os.path.join(args.input_dir, query_file)
    gt_path = os.path.join(args.input_dir, gt_file)
    validate_input_paths([base_path, query_path, gt_path])

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.input_dir, f"deep1b-{args.size}.hdf5")

    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(f"Output exists: {output_path}. Use --force to overwrite.")

    base_mmap, base_count, base_dim = memmap_fbin(base_path, np.float32)
    query = read_fbin(query_path)
    neighbors = read_ibin(gt_path)

    if query.shape[1] != base_dim:
        raise ValueError("Query dimension does not match base dimension")
    if neighbors.shape[0] != query.shape[0]:
        raise ValueError("Ground truth count does not match query count")

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["distance"] = "euclidean"

        chunk_rows = min(args.train_chunk_rows, base_count)
        if chunk_rows <= 0:
            raise ValueError("--train-chunk-rows must be positive")

        chunked_write_train(h5f, base_mmap, chunk_rows)

        h5f.create_dataset("test", data=query, dtype=np.float32)
        write_neighbors_and_distances(h5f, query, base_mmap, neighbors)

    return 0


if __name__ == "__main__":
    sys.exit(main())
