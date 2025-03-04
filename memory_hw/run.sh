#!/bin/bash
# run.sh
# Make sure to chmod +x run.sh
# This script builds and runs both the memory-bound and compute-bound programs
# under various configurations.

echo "==== Building Memory-Bound ===="
nvcc cuda_memory_bound.cu -o cuda_memory_bound
echo "==== Building Compute-Bound ===="
nvcc cuda_compute_bound.cu -o cuda_compute_bound

echo "===================================================================="
echo "Memory-Bound Tests"
echo "===================================================================="

# 1) Small data
echo ""
echo "---- Memory-Bound: 8 blocks, 128 threads, 8 reads, pinned=0 ----"
./cuda_memory_bound 8 128 8 0
echo ""
echo "---- Memory-Bound: 8 blocks, 128 threads, 8 reads, pinned=1 ----"
./cuda_memory_bound 8 128 8 1

# 2) Medium data
echo ""
echo "---- Memory-Bound: 16 blocks, 128 threads, 8 reads, pinned=0 ----"
./cuda_memory_bound 16 128 8 0
echo ""
echo "---- Memory-Bound: 16 blocks, 128 threads, 8 reads, pinned=1 ----"
./cuda_memory_bound 16 128 8 1

# 3) Larger data
echo ""
echo "---- Memory-Bound: 64 blocks, 256 threads, 8 reads, pinned=0 ----"
./cuda_memory_bound 64 256 8 0
echo ""
echo "---- Memory-Bound: 64 blocks, 256 threads, 8 reads, pinned=1 ----"
./cuda_memory_bound 64 256 8 1

echo ""
echo "===================================================================="
echo "Compute-Bound Tests"
echo "===================================================================="

# 1) Fewer ops, pinned=0
echo ""
echo "---- Compute-Bound: 16 blocks, 128 threads, 10000 ops, pinned=0 ----"
./cuda_compute_bound 16 128 10000 0

# 2) Fewer ops, pinned=1
echo ""
echo "---- Compute-Bound: 16 blocks, 128 threads, 10000 ops, pinned=1 ----"
./cuda_compute_bound 16 128 10000 1

# 3) More ops
echo ""
echo "---- Compute-Bound: 16 blocks, 128 threads, 50000 ops, pinned=0 ----"
./cuda_compute_bound 16 128 50000 0

echo ""
echo "---- Compute-Bound: 16 blocks, 128 threads, 50000 ops, pinned=1 ----"
./cuda_compute_bound 16 128 50000 1

# 4) Vary block/thread shape
echo ""
echo "---- Compute-Bound: 8 blocks, 256 threads, 50000 ops, pinned=0 ----"
./cuda_compute_bound 8 256 50000 0

echo ""
echo "---- Compute-Bound: 8 blocks, 256 threads, 50000 ops, pinned=1 ----"
./cuda_compute_bound 8 256 50000 1

echo ""
echo "==== Done! ===="
