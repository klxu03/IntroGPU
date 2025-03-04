/****************************************************************************
 * File: cuda_memory_compare.cu
 *
 * This program demonstrates the same bitwise operation using four
 * different memory approaches:
 *   1) Global Memory: Reads input from global memory.
 *   2) Shared Memory: Loads input from global memory into shared memory,
 *      computes there, and writes back.
 *   3) Constant Memory: Reads input from constant memory.
 *   4) Register Memory: Uses a local register variable for computation.
 *
 * All kernels perform the same looped bitwise operation:
 *      value = ((value ^ CONST_A) | CONST_B) & CONST_C
 * repeated LOOP_COUNT times.
 *
 * Host memory is dynamically allocated (pinned or pageable) based on input.
 *
 * Compile: nvcc cuda_memory_compare.cu -o cuda_memory_compare
 * Run:     ./cuda_memory_compare <numBlocks> <threadsPerBlock> <dataSize> [usePinned=0]
 *
 ****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define LOOP_COUNT     1000
#define MAX_CONST_SIZE 65536  // Maximum size for constant memory input

// ---------------------------------------------------------------------
// Error-checking macro (demonstrates careful host memory management)
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
// Constant memory for bitwise constants and for input (for constant kernel)
// ---------------------------------------------------------------------
__constant__ unsigned int cConstants[3];   // cConstants[0]=CONST_A, [1]=CONST_B, [2]=CONST_C
__constant__ unsigned int cInput[MAX_CONST_SIZE]; // For constant-memory kernel

// Recommended constant values (tunable)
const unsigned int hConstants[3] = { 0x55555555, 0x77777777, 0x33333333 };

// ---------------------------------------------------------------------
// Kernel 1: Global Memory version
// Reads input from global memory and writes result to global memory.
// ---------------------------------------------------------------------
__global__ void kernelGlobalData(const unsigned int *d_in, unsigned int *d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        unsigned int val = d_in[tid]; // 'val' stored in registers
        for (int i = 0; i < LOOP_COUNT; i++) {
            val = (val ^ cConstants[0]) | cConstants[1];
            val = val & cConstants[2];
        }
        d_out[tid] = val;
    }
}

// ---------------------------------------------------------------------
// Kernel 2: Shared Memory version
// Copies data into shared memory, processes it, then writes back.
// ---------------------------------------------------------------------
__global__ void kernelSharedData(const unsigned int *d_in, unsigned int *d_out, int n)
{
    extern __shared__ unsigned int s_data[]; // dynamic shared memory allocation
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int ltid = threadIdx.x; // local thread index

    // Load from global to shared memory
    s_data[ltid] = (tid < n) ? d_in[tid] : 0;
    __syncthreads();

    unsigned int val = s_data[ltid]; // register variable inside shared mem kernel
    for (int i = 0; i < LOOP_COUNT; i++) {
        val = (val ^ cConstants[0]) | cConstants[1];
        val = val & cConstants[2];
    }
    s_data[ltid] = val;
    __syncthreads();

    if (tid < n) {
        d_out[tid] = s_data[ltid];
    }
}

// ---------------------------------------------------------------------
// Kernel 3: Constant Memory version
// Reads input from constant memory and writes result to global memory.
// Note: n must be <= MAX_CONST_SIZE.
// ---------------------------------------------------------------------
__global__ void kernelConstantData(unsigned int *d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        unsigned int val = cInput[tid]; // from constant memory (cached)
        for (int i = 0; i < LOOP_COUNT; i++) {
            val = (val ^ cConstants[0]) | cConstants[1];
            val = val & cConstants[2];
        }
        d_out[tid] = val;
    }
}

// ---------------------------------------------------------------------
// Kernel 4: Register Memory version
// Uses a local variable "result" that is stored in registers for computation.
// ---------------------------------------------------------------------
__global__ void kernelRegisterMemory(const unsigned int *d_in, unsigned int *d_out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n)
    {
        // Local variable "result" is allocated in registers.
        unsigned int result = d_in[tid];
        for (int i = 0; i < LOOP_COUNT; i++) {
            result = (result * 3) + 7;
            result = result ^ 0xAAAAAAAA;
        }
        d_out[tid] = result;
    }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if(argc < 4) {
        fprintf(stderr, "Usage: %s <numBlocks> <threadsPerBlock> <dataSize> [usePinned=0]\n", argv[0]);
        fprintf(stderr, "Example: %s 256 256 65536 1\n", argv[0]);
        return 1;
    }
    int numBlocks       = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int dataSize        = atoi(argv[3]);
    int usePinned       = (argc > 4) ? atoi(argv[4]) : 0;

    if(dataSize > MAX_CONST_SIZE) {
        fprintf(stderr, "For constant memory kernel, dataSize must be <= %d\n", MAX_CONST_SIZE);
        return 1;
    }

    printf("Configuration: %d blocks, %d threads/block, dataSize=%d, usePinned=%d\n",
           numBlocks, threadsPerBlock, dataSize, usePinned);

    // Copy constants to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(cConstants, hConstants, sizeof(hConstants)));

    // ---------------------------------------------------------------------
    // Host Memory Usage:
    // Dynamic allocation of host arrays (using pinned memory if specified)
    // This demonstrates efficient host memory management.
    // ---------------------------------------------------------------------
    unsigned int *h_in = nullptr, *h_out_global = nullptr, *h_out_shared = nullptr;
    unsigned int *h_out_const = nullptr, *h_out_register = nullptr;
    size_t sizeBytes = dataSize * sizeof(unsigned int);

    if(usePinned) {
        printf("Using pinned (page-locked) host memory.\n");
        CUDA_CHECK(cudaMallocHost((void**)&h_in, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_out_global, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_out_shared, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_out_const, sizeBytes));
        CUDA_CHECK(cudaMallocHost((void**)&h_out_register, sizeBytes));
    } else {
        printf("Using regular (pageable) host memory.\n");
        h_in           = new unsigned int[dataSize];
        h_out_global   = new unsigned int[dataSize];
        h_out_shared   = new unsigned int[dataSize];
        h_out_const    = new unsigned int[dataSize];
        h_out_register = new unsigned int[dataSize];
    }

    // Initialize host input data (using same values for consistency)
    for (int i = 0; i < dataSize; i++) {
        h_in[i] = i;
    }

    // ---------------------------------------------------------------------
    // Device memory allocations (for all kernels use the same d_in and d_out arrays)
    // ---------------------------------------------------------------------
    unsigned int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, sizeBytes));

    // Copy host input to device global memory for global/shared/register kernels
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeBytes, cudaMemcpyHostToDevice));
    // For constant kernel, copy host input to constant memory (cInput)
    CUDA_CHECK(cudaMemcpyToSymbol(cInput, h_in, sizeBytes));

    cudaEvent_t start, stop;
    float timeGlobal = 0.0f, timeShared = 0.0f, timeConst = 0.0f, timeRegister = 0.0f;

    // ---------------------------------------------------------------------
    // Kernel 1: Global Memory Kernel
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelGlobalData<<<numBlocks, threadsPerBlock>>>(d_in, d_out, dataSize);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeGlobal, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out_global, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // Kernel 2: Shared Memory Kernel
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    // Allocate shared memory: threadsPerBlock * sizeof(unsigned int)
    kernelSharedData<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(unsigned int)>>>(d_in, d_out, dataSize);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeShared, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out_shared, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // Kernel 3: Constant Memory Kernel
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelConstantData<<<numBlocks, threadsPerBlock>>>(d_out, dataSize);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeConst, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out_const, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // Kernel 4: Register Memory Kernel
    // Demonstrates efficient use of register memory by using a local variable.
    // ---------------------------------------------------------------------
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernelRegisterMemory<<<numBlocks, threadsPerBlock>>>(d_in, d_out, dataSize);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&timeRegister, start, stop));
    CUDA_CHECK(cudaMemcpy(h_out_register, d_out, sizeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // ---------------------------------------------------------------------
    // Print Timing Results and Sample Outputs
    // ---------------------------------------------------------------------
    printf("\n==== Timing Results (ms) ====\n");
    printf("Global Memory Kernel:   %.4f ms\n", timeGlobal);
    printf("Shared Memory Kernel:   %.4f ms\n", timeShared);
    printf("Constant Memory Kernel: %.4f ms\n", timeConst);
    printf("Register Memory Kernel: %.4f ms\n", timeRegister);

    printf("\n==== Sample Outputs (first 5 elements) ====\n");
    printf("Global:   ");
    for (int i = 0; i < 5 && i < dataSize; i++) {
        printf("%u ", h_out_global[i]);
    }
    printf("\nShared:   ");
    for (int i = 0; i < 5 && i < dataSize; i++) {
        printf("%u ", h_out_shared[i]);
    }
    printf("\nConstant: ");
    for (int i = 0; i < 5 && i < dataSize; i++) {
        printf("%u ", h_out_const[i]);
    }
    printf("\nRegister: ");
    for (int i = 0; i < 5 && i < dataSize; i++) {
        printf("%u ", h_out_register[i]);
    }
    printf("\n");

    // ---------------------------------------------------------------------
    // Recommendations for Sample Configurations:
    //
    // 1. Small Data Test (fits in constant memory)
    //    e.g., numBlocks = 32, threadsPerBlock = 256, dataSize = 8192.
    //    Expect constant memory kernel to perform very well.
    //
    // 2. Medium Data Test:
    //    e.g., numBlocks = 256, threadsPerBlock = 256, dataSize = 65536.
    //    All kernels are comparable; constant memory still fits.
    //
    // 3. Larger Data Test (for global, shared, and register kernels)
    //    e.g., numBlocks = 1024, threadsPerBlock = 256, dataSize = 1000000.
    //    Constant kernel cannot be used if dataSize > MAX_CONST_SIZE.
    //
    // 4. Use pinned memory (usePinned=1) to improve host-device transfer performance.
    // ---------------------------------------------------------------------

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    if(usePinned) {
        CUDA_CHECK(cudaFreeHost(h_in));
        CUDA_CHECK(cudaFreeHost(h_out_global));
        CUDA_CHECK(cudaFreeHost(h_out_shared));
        CUDA_CHECK(cudaFreeHost(h_out_const));
        CUDA_CHECK(cudaFreeHost(h_out_register));
    } else {
        delete[] h_in;
        delete[] h_out_global;
        delete[] h_out_shared;
        delete[] h_out_const;
        delete[] h_out_register;
    }

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
