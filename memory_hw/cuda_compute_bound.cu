/****************************************************************************
 * File: cuda_compute_bound.cu
 *
 * This program is analogous to the memory-bound code, but it's more
 * compute-bound. Each thread does repeated bitwise operations in a loop
 * on a small amount of data. We compare four approaches:
 *
 *   1) Global Memory Kernel
 *   2) Shared Memory Kernel
 *   3) Constant Memory Kernel
 *   4) Register Kernel
 *
 * All four kernels do the SAME repeated bitwise transformations on each
 * thread's data, so we can compare performance fairly.
 *
 * The total data size is numBlocks * threadsPerBlock. Each thread processes
 * a single element but does a large number of bitwise ops in a loop to
 * become compute-bound.
 *
 * Usage:
 *   nvcc cuda_compute_bound.cu -o cuda_compute_bound
 *   ./cuda_compute_bound <numBlocks> <threadsPerBlock> <opsPerThread> [usePinned=0]
 *
 * Example:
 *   ./cuda_compute_bound 32 256 100000 1
 ****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAX_CONST_SIZE 16384 // limit for constant memory array

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
__constant__ unsigned int cData[MAX_CONST_SIZE];

// ---------------------------------------------------------------------
// Kernel 1: Global Memory Kernel
// Each thread reads 1 element from global memory, does opsPerThread
// repeated bitwise transformations, and writes back.
// ---------------------------------------------------------------------
__global__ void kernelGlobal(const unsigned int *d_in, unsigned int *d_out, 
                             int opsPerThread, int totalElements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < totalElements) {
        unsigned int val = d_in[tid];
        for (int i = 0; i < opsPerThread; i++) {
            // Some repeated bitwise ops
            val ^= 0x55555555;
            val |= 0x77777777;
            val &= 0x33333333;
            val += 0x11111111;
        }
        d_out[tid] = val;
    }
}

// ---------------------------------------------------------------------
// Kernel 2: Shared Memory Kernel
// Each block loads blockDim.x elements from global into shared memory,
// each thread does repeated ops on its local element in shared mem, then writes back.
// ---------------------------------------------------------------------
__global__ void kernelShared(const unsigned int *d_in, unsigned int *d_out,
                             int opsPerThread, int totalElements)
{
    extern __shared__ unsigned int s_data[]; 
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x;

    if (tid < totalElements) {
        // Load from global into shared
        s_data[ltid] = d_in[tid];
    }
    __syncthreads();

    if (tid < totalElements) {
        unsigned int val = s_data[ltid];
        for (int i = 0; i < opsPerThread; i++) {
            val ^= 0x55555555;
            val |= 0x77777777;
            val &= 0x33333333;
            val += 0x11111111;
        }
        s_data[ltid] = val;
    }
    __syncthreads();

    if (tid < totalElements) {
        d_out[tid] = s_data[ltid];
    }
}

// ---------------------------------------------------------------------
// Kernel 3: Constant Memory Kernel
// Each thread reads from cData[], does repeated ops, writes result to d_out.
// Must ensure totalElements <= MAX_CONST_SIZE.
// ---------------------------------------------------------------------
__global__ void kernelConstant(unsigned int *d_out, int opsPerThread, int totalElements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < totalElements) {
        unsigned int val = cData[tid];
        for (int i = 0; i < opsPerThread; i++) {
            val ^= 0x55555555;
            val |= 0x77777777;
            val &= 0x33333333;
            val += 0x11111111;
        }
        d_out[tid] = val;
    }
}

// ---------------------------------------------------------------------
// Kernel 4: Register Kernel
// Each thread reads from global memory, does repeated ops in a local
// register variable, then writes back. It's the same bitwise ops as the
// other kernels, but we emphasize local register usage for the loop variable.
// ---------------------------------------------------------------------
__global__ void kernelRegister(const unsigned int *d_in, unsigned int *d_out,
                               int opsPerThread, int totalElements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < totalElements) {
        unsigned int val = d_in[tid]; // local register
        for (int i = 0; i < opsPerThread; i++) {
            val ^= 0x55555555;
            val |= 0x77777777;
            val &= 0x33333333;
            val += 0x11111111;
        }
        d_out[tid] = val;
    }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <numBlocks> <threadsPerBlock> <opsPerThread> [usePinned=0]\n", argv[0]);
        fprintf(stderr, "Example: %s 32 256 100000 1\n", argv[0]);
        return 1;
    }
    int numBlocks       = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int opsPerThread    = atoi(argv[3]);
    int usePinned       = (argc > 4) ? atoi(argv[4]) : 0;

    int totalElements = numBlocks * threadsPerBlock;
    printf("Configuration:\n");
    printf("  blocks           = %d\n", numBlocks);
    printf("  threadsPerBlock  = %d\n", threadsPerBlock);
    printf("  opsPerThread     = %d\n", opsPerThread);
    printf("  totalElements    = %d\n", totalElements);
    printf("  usePinned        = %d\n", usePinned);

    if (totalElements > MAX_CONST_SIZE) {
        printf("WARNING: totalElements=%d > MAX_CONST_SIZE=%d, constant kernel won't be valid.\n",
               totalElements, MAX_CONST_SIZE);
    }

    // Host memory
    unsigned int *h_in   = nullptr;
    unsigned int *h_outG = nullptr;
    unsigned int *h_outS = nullptr;
    unsigned int *h_outC = nullptr;
    unsigned int *h_outR = nullptr;

    size_t sizeBytes = totalElements * sizeof(unsigned int);

    if(usePinned) {
        printf("Using pinned (page-locked) host memory.\n");
        CUDA_CHECK(cudaMallocHost((void**)&h_in,   sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outG, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outS, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outC, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_outR, sizeBytes));
    } else {
        printf("Using regular (pageable) host memory.\n");
        h_in   = new unsigned int[totalElements];
        h_outG = new unsigned int[totalElements];
        h_outS = new unsigned int[totalElements];
        h_outC = new unsigned int[totalElements];
        h_outR = new unsigned int[totalElements];
    }

    // Initialize input data
    for(int i = 0; i < totalElements; i++) {
        // random or sequential
        h_in[i] = i % 1024;
    }

    // Device memory
    unsigned int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeBytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeBytes, cudaMemcpyHostToDevice));
    if(totalElements <= MAX_CONST_SIZE) {
        CUDA_CHECK(cudaMemcpyToSymbol(cData, h_in, sizeBytes));
    }

    cudaEvent_t start, stop;
    float msGlobal=0.f, msShared=0.f, msConst=0.f, msRegister=0.f;

    // Kernel 1: Global
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelGlobal<<<numBlocks, threadsPerBlock>>>(d_in, d_out, opsPerThread, totalElements);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msGlobal, start, stop));
    CUDA_CHECK(cudaMemcpy(h_outG, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Kernel 2: Shared
    size_t smemSize = threadsPerBlock * sizeof(unsigned int);
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelShared<<<numBlocks, threadsPerBlock, smemSize>>>(d_in, d_out, opsPerThread, totalElements);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msShared, start, stop));
    CUDA_CHECK(cudaMemcpy(h_outS, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Kernel 3: Constant
    if(totalElements <= MAX_CONST_SIZE) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
        kernelConstant<<<numBlocks, threadsPerBlock>>>(d_out, opsPerThread, totalElements);
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&msConst, start, stop));
        CUDA_CHECK(cudaMemcpy(h_outC, d_out, sizeBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        msConst = -1.f;
    }

    // Kernel 4: Register
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelRegister<<<numBlocks, threadsPerBlock>>>(d_in, d_out, opsPerThread, totalElements);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&msRegister, start, stop));
    CUDA_CHECK(cudaMemcpy(h_outR, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Print timing
    printf("\n=== TIMING (ms) ===\n");
    printf("Global:   %.4f ms\n", msGlobal);
    printf("Shared:   %.4f ms\n", msShared);
    if(msConst >= 0.f) {
        printf("Constant: %.4f ms\n", msConst);
    } else {
        printf("Constant: N/A (too large)\n");
    }
    printf("Register: %.4f ms\n", msRegister);

    // Print sample outputs
    printf("\n=== SAMPLE OUTPUT (first 5 threads) ===\n");
    printf("Global:   ");
    for(int i=0; i<5 && i<totalElements; i++) {
        printf("%u ", h_outG[i]);
    }
    printf("\nShared:   ");
    for(int i=0; i<5 && i<totalElements; i++) {
        printf("%u ", h_outS[i]);
    }
    if(msConst >= 0.f) {
        printf("\nConstant: ");
        for(int i=0; i<5 && i<totalElements; i++) {
            printf("%u ", h_outC[i]);
        }
    }
    printf("\nRegister: ");
    for(int i=0; i<5 && i<totalElements; i++) {
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
