#!/bin/bash
# run.sh
# 
# This script compiles the memory-bound and compute-bound CUDA programs,
# then runs them with various block/thread/pinned configurations.
# We test 4 block sizes, 4 thread-per-block sizes, and pinned=0 or 1.
# 
# For the memory-bound code, we fix readsPerThread=8.
# For the compute-bound code, we fix opsPerThread=50000.
#
# Adjust as needed for your assignment or system.

# 1) Compile both programs
echo "==== Building Memory-Bound ===="
nvcc cuda_memory_bound.cu -o cuda_memory_bound
echo "==== Building Compute-Bound ===="
nvcc cuda_compute_bound.cu -o cuda_compute_bound

# 2) Define parameter sets
blocksList=(4 16 64 128)
threadsList=(64 128 256 512)
pinnedList=(0 1)

READS=8      # memory-bound setting
OPS=50000    # compute-bound setting

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
echo "Compute-Bound Tests (opsPerThread=$OPS)"
echo "===================================================================="
for b in "${blocksList[@]}"; do
  for t in "${threadsList[@]}"; do
    for p in "${pinnedList[@]}"; do
      echo ""
      echo "---- Compute-Bound: $b blocks, $t threads, $OPS ops, pinned=$p ----"
      ./cuda_compute_bound $b $t $OPS $p
    done
  done
done

echo ""
echo "==== Done! ===="
