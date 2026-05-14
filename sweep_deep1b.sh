#!/bin/bash

set -e

NUM_VECTORS=(
    10
    100
    1000
    10000
    100000
    1000000
    10000000
    100000000
    1000000000
)

for n in "${NUM_VECTORS[@]}"; do
    python create_deep1b.py --n $n --output $SCRATCH/data/deep1b/deep1b-n$n.hdf5
done
