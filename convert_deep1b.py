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


def write_neighbors_and_distances(
    h5f,
    query,
    base_mmap,
    neighbors,
    batch_rows,
    start_row,
    neighbors_ds,
    distances_ds,
    checkpoint_attr,
):
    num_queries = query.shape[0]
    top_k = neighbors.shape[1]
    dim = base_mmap.shape[1]

    oob_count = 0
    oob_queries = 0
    for start in tqdm(
        range(start_row, num_queries, batch_rows),
        desc="Writing /neighbors and /distances",
    ):
        end = min(start + batch_rows, num_queries)
        batch_query = query[start:end]
        batch_neighbors = neighbors[start:end].astype(np.int32, copy=False)
        valid_mask = batch_neighbors < base_mmap.shape[0]
        oob_in_batch = (~valid_mask).sum(axis=1)
        oob_queries += int((oob_in_batch > 0).sum())
        oob_count += int(oob_in_batch.sum())

        safe_neighbors = batch_neighbors.copy()
        safe_neighbors[~valid_mask] = 0
        neighbor_vectors = base_mmap[safe_neighbors.reshape(-1)].reshape(
            end - start, top_k, dim
        )
        diff = neighbor_vectors - batch_query[:, None, :]
        distances = np.einsum("bij,bij->bi", diff, diff, dtype=np.float32)

        neighbors_out = batch_neighbors.copy()
        neighbors_out[~valid_mask] = -1
        distances_out = distances.astype(np.float32, copy=False)
        distances_out[~valid_mask] = np.inf

        neighbors_ds[start:end] = neighbors_out
        distances_ds[start:end] = distances_out

        h5f.attrs[checkpoint_attr] = end
        h5f.flush()

    if oob_count:
        print(
            f"WARN: {oob_queries} queries had {oob_count} neighbors out of bounds (set to -1).",
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
        default=50000,
        help="Rows per chunk for /train dataset",
    )
    parser.add_argument(
        "--neighbor-batch-rows",
        type=int,
        default=1024,
        help="Rows per batch for /neighbors and /distances",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a partial run")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing HDF5 output and exit",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if exists")
    return parser.parse_args()


def validate_hdf5(output_path, base_count, base_dim, num_queries, top_k):
    with h5py.File(output_path, "r") as h5f:
        if "distance" not in h5f.attrs:
            raise ValueError("Missing distance attribute")
        if h5f.attrs["distance"] != "euclidean":
            raise ValueError("distance attribute must be 'euclidean'")

        for name in ("train", "test", "neighbors", "distances"):
            if name not in h5f:
                raise ValueError(f"Missing dataset /{name}")

        train = h5f["train"]
        test = h5f["test"]
        neighbors = h5f["neighbors"]
        distances = h5f["distances"]

        if train.ndim != 2:
            raise ValueError("Expected /train to be rank-2")
        if test.ndim != 2:
            raise ValueError("Expected /test to be rank-2")
        if neighbors.ndim != 2:
            raise ValueError("Expected /neighbors to be rank-2")
        if distances.ndim != 2:
            raise ValueError("Expected /distances to be rank-2")

        if train.dtype != np.float32:
            raise ValueError("Expected /train dtype float32")
        if test.dtype != np.float32:
            raise ValueError("Expected /test dtype float32")
        if neighbors.dtype != np.int32:
            raise ValueError("Expected /neighbors dtype int32")
        if distances.dtype != np.float32:
            raise ValueError("Expected /distances dtype float32")

        if train.shape != (base_count, base_dim):
            raise ValueError("/train shape mismatch")
        if test.shape != (num_queries, base_dim):
            raise ValueError("/test shape mismatch")
        if neighbors.shape != (num_queries, top_k):
            raise ValueError("/neighbors shape mismatch")
        if distances.shape != (num_queries, top_k):
            raise ValueError("/distances shape mismatch")


def validate_checkpoint_attr(h5f, key, max_value):
    if key not in h5f.attrs:
        return 0
    value = int(h5f.attrs[key])
    if value < 0 or value > max_value:
        raise ValueError(f"Invalid {key} checkpoint value")
    return value


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

    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    if args.force and args.validate_only:
        raise ValueError("--force and --validate-only are mutually exclusive")

    output_exists = os.path.exists(output_path)
    if args.resume and not output_exists:
        raise FileNotFoundError(f"Output not found for resume: {output_path}")
    if args.validate_only and not output_exists:
        raise FileNotFoundError(f"Output not found for validation: {output_path}")
    if output_exists and not (args.force or args.resume or args.validate_only):
        raise FileExistsError(f"Output exists: {output_path}. Use --force or --resume.")

    base_mmap, base_count, base_dim = memmap_fbin(base_path, np.float32)
    query = read_fbin(query_path)
    neighbors = read_ibin(gt_path)

    if query.shape[1] != base_dim:
        raise ValueError("Query dimension does not match base dimension")
    if neighbors.shape[0] != query.shape[0]:
        raise ValueError("Ground truth count does not match query count")
    top_k = neighbors.shape[1]

    if args.validate_only:
        validate_hdf5(output_path, base_count, base_dim, query.shape[0], top_k)
        print(f"Validation OK: {output_path}")
        return 0

    file_mode = "a" if args.resume else "w"
    with h5py.File(output_path, file_mode) as h5f:
        if args.resume:
            validate_hdf5(output_path, base_count, base_dim, query.shape[0], top_k)
            train_ds = h5f["train"]
            test_ds = h5f["test"]
            neighbors_ds = h5f["neighbors"]
            distances_ds = h5f["distances"]
        else:
            h5f.attrs["distance"] = "euclidean"
            train_ds = h5f.create_dataset(
                "train",
                shape=(base_count, base_dim),
                dtype=np.float32,
                chunks=(min(args.train_chunk_rows, base_count), base_dim),
            )
            test_ds = h5f.create_dataset("test", data=query, dtype=np.float32)
            neighbors_ds = h5f.create_dataset(
                "neighbors",
                shape=(query.shape[0], top_k),
                dtype=np.int32,
            )
            distances_ds = h5f.create_dataset(
                "distances",
                shape=(query.shape[0], top_k),
                dtype=np.float32,
            )

        chunk_rows = min(args.train_chunk_rows, base_count)
        if chunk_rows <= 0:
            raise ValueError("--train-chunk-rows must be positive")

        train_start = validate_checkpoint_attr(h5f, "train_rows_written", base_count)
        if train_start < base_count:
            for start in tqdm(
                range(train_start, base_count, chunk_rows),
                desc="Writing /train",
            ):
                end = min(start + chunk_rows, base_count)
                train_ds[start:end] = base_mmap[start:end]
                h5f.attrs["train_rows_written"] = end
                h5f.flush()

        if test_ds.shape != query.shape:
            raise ValueError("/test shape mismatch for resume")

        neighbor_batch_rows = args.neighbor_batch_rows
        if neighbor_batch_rows <= 0:
            raise ValueError("--neighbor-batch-rows must be positive")
        neighbors_start = validate_checkpoint_attr(
            h5f, "neighbors_rows_written", query.shape[0]
        )
        if neighbors_start < query.shape[0]:
            write_neighbors_and_distances(
                h5f,
                query,
                base_mmap,
                neighbors,
                neighbor_batch_rows,
                neighbors_start,
                neighbors_ds,
                distances_ds,
                "neighbors_rows_written",
            )

        if "train_rows_written" in h5f.attrs:
            del h5f.attrs["train_rows_written"]
        if "neighbors_rows_written" in h5f.attrs:
            del h5f.attrs["neighbors_rows_written"]

    return 0


if __name__ == "__main__":
    sys.exit(main())
