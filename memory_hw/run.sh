#!/bin/bash
# run.sh
# 
# This script compiles the memory-bound CUDA program,
# then runs it with various block/thread/pinned configurations.
# For the memory-bound code, we fix readsPerThread=8.

# 1) Compile program
echo "==== Building Memory-Bound ===="
nvcc cuda_memory_bound.cu -o cuda_memory_bound

# 2) Define parameter sets
blocksList=(4 16 64 128)
threadsList=(64 128 256 512)
pinnedList=(0 1)

READS=8

echo ""
echo "===================================================================="
echo "Memory-Bound Tests (readsPerThread=$READS)"
echo "===================================================================="
for b in "${blocksList[@]}"; do
  for t in "${threadsList[@]}"; do
    for p in "${pinnedList[@]}"; do
      echo ""
      echo "---- Memory-Bound: $b blocks, $t threads, $READS reads, pinned=$p ----"
      ./cuda_memory_bound $b $t $READS $p
    done
  done
done

echo ""
echo "===================================================================="
echo "Large Data Copy Test - Pinned vs. Non-Pinned"
echo "===================================================================="
echo ""
echo "---- Memory-Bound: 512 blocks, 512 threads, 2000 reads, pinned=0 ----"
./cuda_memory_bound 512 512 2000 0

echo ""
echo "---- Memory-Bound: 512 blocks, 512 threads, 2000 reads, pinned=1 ----"
./cuda_memory_bound 512 512 2000 1

echo ""
echo "==== Done! ===="