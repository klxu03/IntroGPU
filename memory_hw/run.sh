#!/bin/bash
# run.sh
# 
# This script compiles the memory-bound CUDA program,
# then runs it with various block/thread/pinned configurations.
# We test 4 block sizes, 4 thread-per-block sizes, and pinned=0 or 1.
# 
# For the memory-bound code, we fix readsPerThread=8.
#
# Adjust as needed for your assignment or system.

# 1) Compile both programs
echo "==== Building Memory-Bound ===="
nvcc cuda_memory_bound.cu -o cuda_memory_bound

# 2) Define parameter sets
blocksList=(4 16 64 128)
threadsList=(64 128 256 512)
pinnedList=(0 1)

READS=8      # memory-bound setting

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
echo "---- Memory-Bound: 512 blocks, 512 threads, 10000 reads, pinned=0 ----"
./cuda_memory_bound 512 512 10000 0

echo ""
echo "---- Memory-Bound: 512 blocks, 512 threads, 10000 reads, pinned=1 ----"
./cuda_memory_bound 512 512 10000 1

echo ""
echo "==== Done! ===="
