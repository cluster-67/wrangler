# Plan: Convert Yandex Deep1B Dataset to HDF5 Format

## Goal
Convert Yandex Deep1B dataset files (.fbin and .ibin) to HDF5 format matching the ann-benchmarks structure shown in output-format.md.

## Scope (Confirmed)
- Dataset: Deep1B only
- Sizes: 10M and 1B
- Ignore learn.350M (no ground truth)
- Output files: `deep1b-10M.hdf5`, `deep1b-1B.hdf5`
- Distance attribute: `"euclidean"`
- Distances stored: squared L2

## Source Files (Deep1B)
From https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search:

Shared across sizes:
- Query vectors: `query.public.10K.fbin`
- Ground truth: `groundtruth.public.10K.ibin`

Per size:
- Database vectors: `base.10M.fbin`, `base.1B.fbin`

## Target HDF5 Structure
Based on output-format.md:
```
/                        Group
    Attribute: distance scalar (string: "euclidean")
/distances               Dataset {num_queries/num_queries, 100/100} (float)
/neighbors               Dataset {num_queries/num_queries, 100/100} (int)
/test                    Dataset {num_queries/num_queries, vector_dim/vector_dim} (float)
/train                   Dataset {num_database/num_database, vector_dim/vector_dim} (float)
```

## Data Format Specifications
- **.fbin**: [num_vectors (uint32), vector_dim (uint32), vector_array (float32)]
- **.ibin**: [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]

## Environment Setup
- Load Python: `module load python/3.12-26.1.0`
- Create venv and install deps:
  - `python -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install numpy h5py tqdm`
- For HDF5 inspection (later): `module load cray-hdf5`

## Download Script
Create `download_deep1b.sh`:
- Target dir: `$SCRATCH/data/deep1b/`
- Use `wget -c -nc` so downloads resume and skip if file exists
- Files:
  - `base.10M.fbin`
  - `base.1B.fbin`
  - `query.public.10K.fbin`
  - `groundtruth.public.10K.ibin`
- No file name clashes; query/gt are shared and only one copy is needed.

## Conversion Script
Create `convert_deep1b.py`:
- CLI flags:
  - `--size {10M,1B}`
  - `--input-dir` (default `$SCRATCH/data/deep1b`)
  - `--output` (default `$SCRATCH/data/deep1b/deep1b-{size}.hdf5`)
  - `--batch-size` (for distance computation)
  - `--force` (overwrite existing HDF5)
- Default behavior: refuse to overwrite unless `--force` is provided.

### Memory/Performance Strategy (Ideal Solution)
- Use `numpy.memmap` for base vectors to avoid full RAM load.
- Load query + ground truth fully (small).
- Compute distances in batches using vectorized numpy L2.
- Write `/train` in chunks from memmap; `/test`, `/neighbors`, `/distances` in normal writes.

### Distance Computation (Squared L2)
- For each query i and its neighbor indices:
  - Gather neighbor vectors from base memmap
  - Compute squared L2 via vectorized numpy
  - Store into `/distances` as float32

## Execution Steps
1. Run `download_deep1b.sh`
2. Convert 10M:
   - `python convert_deep1b.py --size 10M`
3. Convert 1B:
   - `python convert_deep1b.py --size 1B`
4. Validate structure with `h5ls` / `h5dump` (after `module load cray-hdf5`)
