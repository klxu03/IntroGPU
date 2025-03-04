/****************************************************************************
 * File: cuda_memory_bound.cu
 *
 * Demonstrates a memory-bound scenario where each kernel performs
 * the SAME summation of multiple elements, but uses different memory types:
 *
 *   1) Global Memory Kernel
 *   2) Shared Memory Kernel
 *   3) Constant Memory Kernel
 *   4) Register Kernel (still must read from memory, but uses registers
 *      to accumulate sums).
 *
 * Each thread sums READS_PER_THREAD elements from an array and stores
 * the result in d_out[tid].
 *
 * By design, there's minimal arithmetic but lots of memory reads, so
 * the performance depends heavily on memory throughput and caching.
 *
 * USAGE:
 *   nvcc cuda_memory_bound.cu -o cuda_memory_bound
 *   ./cuda_memory_bound <numBlocks> <threadsPerBlock> <readsPerThread> [usePinned=0]
 *
 *   - dataSize is computed as: dataSize = numBlocks * threadsPerBlock * readsPerThread
 *   - If dataSize > MAX_CONST_SIZE, the constant kernel won't work.
 *
 * Example:
 *   ./cuda_memory_bound 32 256 8 1
 ****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAX_CONST_SIZE 16384  // Must be <= 16384 so each element fits in constant memory

// ---------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(err);                                                       \
    }                                                                    \
} while(0)

// ---------------------------------------------------------------------
// Constant memory array
// ---------------------------------------------------------------------
__constant__ unsigned int cInput[MAX_CONST_SIZE];

// ---------------------------------------------------------------------
// Kernel 1: Global Memory
// Each thread reads READS_PER_THREAD elements from global memory d_in.
// Summation is minimal arithmetic, highlighting memory usage.
// ---------------------------------------------------------------------
__global__ void kernelGlobal(const unsigned int *d_in, unsigned int *d_out, 
                             int readsPerThread, int totalReads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < totalReads) {
        unsigned int sum = 0;
        // The chunk of elements that this thread processes
        int baseIdx = tid * readsPerThread;
        for(int i = 0; i < readsPerThread; i++) {
            sum += d_in[baseIdx + i];
        }
        d_out[tid] = sum;
    }
}

// ---------------------------------------------------------------------
// Kernel 2: Shared Memory
// Each block loads a chunk of data from global into shared memory, 
// then each thread reads from shared memory multiple times to sum up.
// ---------------------------------------------------------------------
__global__ void kernelShared(const unsigned int *d_in, unsigned int *d_out,
                             int readsPerThread, int totalReads)
{
    extern __shared__ unsigned int s_data[];
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int ltid  = threadIdx.x;  // local thread ID
    if(tid < totalReads) {
        // Each block processes blockDim.x * readsPerThread elements
        int blockSize = blockDim.x * readsPerThread;
        int blockStart = blockIdx.x * blockSize;

        // Copy that block’s chunk from global -> shared
        for(int i = ltid; i < blockSize; i += blockDim.x) {
            s_data[i] = d_in[blockStart + i];
        }
        __syncthreads();

        // Now each thread sums the portion from shared
        unsigned int sum = 0;
        int baseIdx = ltid * readsPerThread;
        for(int i = 0; i < readsPerThread; i++) {
            sum += s_data[baseIdx + i];
        }
        d_out[tid] = sum;
    }
}

// ---------------------------------------------------------------------
// Kernel 3: Constant Memory
// Each thread reads from cInput[] (cached constant memory).
// ---------------------------------------------------------------------
__global__ void kernelConstant(unsigned int *d_out, int readsPerThread, int totalReads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < totalReads) {
        unsigned int sum = 0;
        int baseIdx = tid * readsPerThread;
        for(int i = 0; i < readsPerThread; i++) {
            sum += cInput[baseIdx + i];
        }
        d_out[tid] = sum;
    }
}

// ---------------------------------------------------------------------
// Kernel 4: Register Kernel
// We still have to read from global memory, but we accumulate 
// everything in a local register variable. 
// (It's the same summation, but we highlight that the final 
// accumulation is purely in registers.)
// ---------------------------------------------------------------------
__global__ void kernelRegister(const unsigned int *d_in, unsigned int *d_out,
                               int readsPerThread, int totalReads)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < totalReads) {
        unsigned int sumReg = 0; // This variable stays in registers
        int baseIdx = tid * readsPerThread;
        for(int i = 0; i < readsPerThread; i++) {
            sumReg += d_in[baseIdx + i];
        }
        d_out[tid] = sumReg;
    }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(argc < 4) {
        fprintf(stderr, "Usage: %s <numBlocks> <threadsPerBlock> <readsPerThread> [usePinned=0]\n", argv[0]);
        fprintf(stderr, "Example: %s 32 256 8 1\n", argv[0]);
        return 1;
    }

    int numBlocks       = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int readsPerThread  = atoi(argv[3]);
    int usePinned       = (argc > 4) ? atoi(argv[4]) : 0;

    // totalReads = number of threads total
    int totalThreads = numBlocks * threadsPerBlock;
    // total data size = totalThreads * readsPerThread
    int dataSize = totalThreads * readsPerThread;

    printf("Configuration:\n");
    printf("  blocks            = %d\n", numBlocks);
    printf("  threadsPerBlock   = %d\n", threadsPerBlock);
    printf("  readsPerThread    = %d\n", readsPerThread);
    printf("  totalThreads      = %d\n", totalThreads);
    printf("  dataSize          = %d\n", dataSize);
    printf("  usePinned         = %d\n", usePinned);

    // If dataSize is bigger than MAX_CONST_SIZE, constant kernel won't work
    if(dataSize > MAX_CONST_SIZE) {
        printf("WARNING: dataSize=%d > MAX_CONST_SIZE=%d, constant kernel won't be valid.\n",
               dataSize, MAX_CONST_SIZE);
    }

    // Host memory
    unsigned int *h_in   = nullptr;
    unsigned int *h_outG = nullptr;  // global
    unsigned int *h_outS = nullptr;  // shared
    unsigned int *h_outC = nullptr;  // constant
    unsigned int *h_outR = nullptr;  // register

    size_t sizeBytes = dataSize * sizeof(unsigned int);
    size_t outBytes  = totalThreads * sizeof(unsigned int);

    if(usePinned) {
        printf("Using pinned (page-locked) host memory.\n");
        CUDA_CHECK(cudaMallocHost((void**)&h_in,   sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outG, outBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outS, outBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outC, outBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outR, outBytes));
    } else {
        printf("Using regular (pageable) host memory.\n");
        h_in   = new unsigned int[dataSize];
        h_outG = new unsigned int[totalThreads];
        h_outS = new unsigned int[totalThreads];
        h_outC = new unsigned int[totalThreads];
        h_outR = new unsigned int[totalThreads];
    }

    // Initialize input data
    for(int i = 0; i < dataSize; i++) {
        // random or sequential
        h_in[i] = i % 1024;
    }

    // Device memory
    unsigned int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, outBytes));

    // Copy host->device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeBytes, cudaMemcpyHostToDevice));
    // Copy to constant if within limit
    if(dataSize <= MAX_CONST_SIZE) {
        CUDA_CHECK(cudaMemcpyToSymbol(cInput, h_in, sizeBytes));
    }

    // Timers
    cudaEvent_t start, stop;
    float msGlobal   = 0.f;
    float msShared   = 0.f;
    float msConstant = 0.f;
    float msRegister = 0.f;

    // ---------------------------------------------------------------------
    // 1) Global Kernel
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelGlobal<<<numBlocks, threadsPerBlock>>>(d_in, d_out, readsPerThread, totalThreads);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msGlobal, start, stop));
    CUDA_CHECK(cudaMemcpy(h_outG, d_out, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // 2) Shared Kernel
    // Each block needs blockDim.x * readsPerThread * sizeof(unsigned int) in shared mem
    // ---------------------------------------------------------------------
    size_t smemSize = threadsPerBlock * readsPerThread * sizeof(unsigned int);
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelShared<<<numBlocks, threadsPerBlock, smemSize>>>(d_in, d_out, readsPerThread, totalThreads);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msShared, start, stop));
    CUDA_CHECK(cudaMemcpy(h_outS, d_out, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // 3) Constant Kernel
    // Only run if dataSize <= MAX_CONST_SIZE
    // ---------------------------------------------------------------------
    if(dataSize <= MAX_CONST_SIZE) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
        kernelConstant<<<numBlocks, threadsPerBlock>>>(d_out, readsPerThread, totalThreads);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&msConstant, start, stop));
        CUDA_CHECK(cudaMemcpy(h_outC, d_out, outBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        msConstant = -1.f; // not valid
    }

    // ---------------------------------------------------------------------
    // 4) Register Kernel
    // It's the same summation, but we highlight that the sum variable
    // is stored in registers. We still read from global memory.
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelRegister<<<numBlocks, threadsPerBlock>>>(d_in, d_out, readsPerThread, totalThreads);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msRegister, start, stop));
    CUDA_CHECK(cudaMemcpy(h_outR, d_out, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // Print results
    // ---------------------------------------------------------------------
    printf("\n=== TIMING (ms) ===\n");
    printf("Global:   %.4f ms\n",   msGlobal);
    printf("Shared:   %.4f ms\n",   msShared);
    if(msConstant >= 0.f) {
        printf("Constant: %.4f ms\n", msConstant);
    } else {
        printf("Constant: N/A (data too large)\n");
    }
    printf("Register: %.4f ms\n",   msRegister);

    printf("\n=== SAMPLE OUTPUT (first 5 threads) ===\n");
    printf("Global:   "); 
    for(int i = 0; i < 5 && i < totalThreads; i++) {
        printf("%u ", h_outG[i]);
    }
    printf("\nShared:   ");
    for(int i = 0; i < 5 && i < totalThreads; i++) {
        printf("%u ", h_outS[i]);
    }
    if(msConstant >= 0.f) {
        printf("\nConstant: ");
        for(int i = 0; i < 5 && i < totalThreads; i++) {
            printf("%u ", h_outC[i]);
        }
    }
    printf("\nRegister: ");
    for(int i = 0; i < 5 && i < totalThreads; i++) {
        printf("%u ", h_outR[i]);
    }
    printf("\n\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    if(usePinned) {
        CUDA_CHECK(cudaFreeHost(h_in));
        CUDA_CHECK(cudaFreeHost(h_outG));
        CUDA_CHECK(cudaFreeHost(h_outS));
        CUDA_CHECK(cudaFreeHost(h_outC));
        CUDA_CHECK(cudaFreeHost(h_outR));
    } else {
        delete[] h_in;
        delete[] h_outG;
        delete[] h_outS;
        delete[] h_outC;
        delete[] h_outR;
    }

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
