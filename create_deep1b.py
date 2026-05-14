#!/usr/bin/env python3
"""
Build a Deep1B HDF5 subset: first ``n`` vectors from ``base.1B.fbin``, public 10K
queries, and exact k-nearest neighbors (Euclidean) via sklearn.

Memory (float32 train matrix) is about ``n * dim * 4`` bytes plus index overhead.
Exact ``kneighbors`` cost scales roughly with ``num_queries * n * dim`` for
brute-force paths; very large ``n`` can take a long time.

Parallelism: use ``--n-jobs`` (default all cores). With ``n_jobs != 1``, BLAS
threads are capped via ``threadpoolctl`` when available to avoid oversubscription;
otherwise set ``OMP_NUM_THREADS=1`` (and MKL/OpenBLAS equivalents) for best throughput.

Examples::

    python create_deep1b.py --n 1000000
    python create_deep1b.py --n 10000000 --output /path/to/deep1b-n10M.hdf5
    python create_deep1b.py --n 100000000 --n-jobs 64 --neighbor-batch-rows 2048
    python create_deep1b.py --n 1000000000 --train-chunk-rows 100000
"""
import argparse
import contextlib
import inspect
import os
import sys

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from convert_deep1b import memmap_fbin, read_fbin, validate_hdf5, validate_input_paths


try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def _threadpool_or_warn(n_jobs: int):
    if n_jobs == 1:
        return contextlib.nullcontext()
    if threadpool_limits is not None:
        return threadpool_limits(limits=1)
    print(
        "WARN: threadpoolctl not installed; with n_jobs!=1 set OMP_NUM_THREADS=1, "
        "MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 to avoid BLAS oversubscription.",
        file=sys.stderr,
    )
    return contextlib.nullcontext()


def _kneighbors_kwargs(n_jobs: int) -> dict:
    sig = inspect.signature(NearestNeighbors.kneighbors)
    if "n_jobs" in sig.parameters:
        return {"n_jobs": n_jobs}
    return {}


def parse_args():
    default_input = os.path.join(os.environ.get("SCRATCH", ""), "data", "deep1b")
    parser = argparse.ArgumentParser(
        description=(
            "Create Deep1B HDF5 from the first n base points (base.1B.fbin prefix), "
            "public 10K queries, and exact sklearn k-NN."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of base points to read from the start of base.1B.fbin",
    )
    parser.add_argument(
        "--input-dir",
        default=default_input,
        help="Directory containing base.1B.fbin and query.public.10K.fbin",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HDF5 (default: INPUT_DIR/deep1b-n{N}.hdf5)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of neighbors per query (default: 100)",
    )
    parser.add_argument(
        "--train-chunk-rows",
        type=int,
        default=50000,
        help="Rows per chunk when writing /train to HDF5",
    )
    parser.add_argument(
        "--neighbor-batch-rows",
        type=int,
        default=None,
        help=(
            "If set, run kneighbors and write /neighbors and /distances in row "
            "batches of this size (queries are still held in memory)"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for NearestNeighbors / kneighbors (default: -1 = all cores)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_dir:
        raise ValueError("--input-dir must be set or SCRATCH must be defined")

    n = args.n
    k = args.k
    if n <= 0:
        raise ValueError("--n must be positive")
    if k <= 0:
        raise ValueError("--k must be positive")
    if n < k:
        raise ValueError(f"Need n >= k for fixed output width; got n={n}, k={k}")

    base_path = os.path.join(args.input_dir, "base.1B.fbin")
    query_path = os.path.join(args.input_dir, "query.public.10K.fbin")
    validate_input_paths([base_path, query_path])

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.input_dir, f"deep1b-n{n}.hdf5")

    if os.path.exists(output_path) and not args.force:
        raise FileExistsError(f"Output exists: {output_path}. Use --force to overwrite.")

    base_mmap, base_count, base_dim = memmap_fbin(base_path, np.float32)
    if n > base_count:
        raise ValueError(f"--n={n} exceeds base file count {base_count}")

    print(f"Loading {n} x {base_dim} base vectors into memory...", flush=True)
    train = np.ascontiguousarray(base_mmap[:n], dtype=np.float32)

    test = read_fbin(query_path)
    if test.shape[1] != base_dim:
        raise ValueError("Query dimension does not match base dimension")

    num_queries = test.shape[0]
    bytes_train = train.nbytes
    print(
        f"Train array ~{bytes_train / (1024**3):.2f} GiB; {num_queries} queries, k={k}.",
        flush=True,
    )

    chunk_rows = min(args.train_chunk_rows, n)
    if chunk_rows <= 0:
        raise ValueError("--train-chunk-rows must be positive")

    neighbor_batch = args.neighbor_batch_rows
    if neighbor_batch is not None and neighbor_batch <= 0:
        raise ValueError("--neighbor-batch-rows must be positive when set")

    kneighbor_kw = _kneighbors_kwargs(args.n_jobs)

    print("Fitting NearestNeighbors...", flush=True)
    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=args.n_jobs,
    )
    with _threadpool_or_warn(args.n_jobs):
        nn.fit(train)

    indices_out = np.empty((num_queries, k), dtype=np.int32)
    distances_out = np.empty((num_queries, k), dtype=np.float32)

    if neighbor_batch is None:
        print("Running kneighbors for all queries...", flush=True)
        with _threadpool_or_warn(args.n_jobs):
            dist, idx = nn.kneighbors(
                test, n_neighbors=k, return_distance=True, **kneighbor_kw
            )
        distances_out[:] = dist.astype(np.float32, copy=False)
        indices_out[:] = idx.astype(np.int32, copy=False)
    else:
        print(
            f"Running kneighbors in query batches of {neighbor_batch} rows...",
            flush=True,
        )
        for start in tqdm(
            range(0, num_queries, neighbor_batch),
            desc="kneighbors",
        ):
            end = min(start + neighbor_batch, num_queries)
            batch = test[start:end]
            with _threadpool_or_warn(args.n_jobs):
                dist, idx = nn.kneighbors(
                    batch, n_neighbors=k, return_distance=True, **kneighbor_kw
                )
            distances_out[start:end] = dist.astype(np.float32, copy=False)
            indices_out[start:end] = idx.astype(np.int32, copy=False)

    print(f"Writing HDF5: {output_path}", flush=True)
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["distance"] = "euclidean"
        train_ds = h5f.create_dataset(
            "train",
            shape=(n, base_dim),
            dtype=np.float32,
            chunks=(min(chunk_rows, n), base_dim),
        )
        h5f.create_dataset("test", data=test, dtype=np.float32)
        h5f.create_dataset("neighbors", data=indices_out, dtype=np.int32)
        h5f.create_dataset("distances", data=distances_out, dtype=np.float32)

        for start in tqdm(range(0, n, chunk_rows), desc="Writing /train"):
            end = min(start + chunk_rows, n)
            train_ds[start:end] = train[start:end]
            h5f.flush()

    validate_hdf5(output_path, n, base_dim, num_queries, k)
    print(f"Done. Validation OK: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
