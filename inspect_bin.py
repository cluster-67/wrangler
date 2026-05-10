#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np


HEADER_DTYPE = np.dtype([("count", "<u4"), ("dim", "<u4")])


def read_header(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=HEADER_DTYPE, count=1)
    if header.size != 1:
        raise ValueError(f"Failed to read header from {path}")
    return int(header[0]["count"]), int(header[0]["dim"])


def main():
    parser = argparse.ArgumentParser(
        description="Inspect .fbin/.ibin files (count, dim, size checks)"
    )
    parser.add_argument("path", help="Path to .fbin or .ibin file")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".fbin", ".ibin"}:
        raise ValueError("Expected a .fbin or .ibin file")

    dtype = np.float32 if ext == ".fbin" else np.int32
    count, dim = read_header(path)
    actual_bytes = os.path.getsize(path)
    expected_bytes = HEADER_DTYPE.itemsize + (count * dim * np.dtype(dtype).itemsize)

    print(f"path: {path}")
    print(f"type: {ext[1:]} ({dtype})")
    print(f"vectors: {count}")
    print(f"dimension: {dim}")
    print(f"header_bytes: {HEADER_DTYPE.itemsize}")
    print(f"expected_bytes: {expected_bytes}")
    print(f"actual_bytes: {actual_bytes}")
    if actual_bytes < expected_bytes:
        print("status: file too small", file=sys.stderr)
        return 2
    if actual_bytes > expected_bytes:
        print("status: file larger than expected", file=sys.stderr)
        return 3
    print("status: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
