/****************************************************************************
 * File: single_chunk.cu
 *
 * Demonstrates a single-stream (non-pipelined) CUDA program that processes 
 * all data at once (in a single chunk). It:
 *   1) Copies data & weights from host to device.
 *   2) Multiplies them.
 *   3) Reduces both arrays to a single sum each.
 *   4) Copies those partial sums back.
 * Finally, it computes the weighted average and prints the total timing.
 *
 * Usage:
 *   nvcc single_chunk.cu -o single_chunk
 *   ./single_chunk <num_elements> <threads_per_block>
 *
 * Example:
 *   ./single_chunk 1000000 256
 ****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                             \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));             \
        exit(err);                                                        \
    }                                                                     \
} while(0)

// ---------------------------------------------------------------------
// KERNELS
// ---------------------------------------------------------------------

// multiplyKernel: element-wise product -> d_prod[i] = d_data[i] * d_weights[i]
__global__ void multiplyKernel(const float* d_data, const float* d_weights, 
                               float* d_prod, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_prod[idx] = d_data[idx] * d_weights[idx];
}

// reductionKernel: sums an array in shared memory, writing one partial sum per block.
__global__ void reductionKernel(const float* d_in, float* d_out, int n)
{
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // sum the pair of elements this thread is responsible for
    float sum = 0.0f;
    if (idx < n) {
        sum = d_in[idx];
    }
    if (idx + blockDim.x < n) {
        sum += d_in[idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    // reduce partial sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

// ---------------------------------------------------------------------
// HELPER FUNCTIONS
// ---------------------------------------------------------------------

// performReductionAsync: repeatedly calls reductionKernel until only one float remains
static void performReductionAsync(float* d_input, int n, int threadsPerBlock,
                                  cudaStream_t stream, float** d_result)
{
    int currElements = n;
    float* d_in = d_input;
    float* d_out = NULL;

    while (currElements > 1) {
        int blocks = (currElements + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        size_t smemSize = threadsPerBlock * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_out, blocks * sizeof(float)));
        reductionKernel<<<blocks, threadsPerBlock, smemSize, stream>>>(d_in, d_out, currElements);
        CUDA_CHECK(cudaGetLastError());
        if (d_in != d_input) {
            CUDA_CHECK(cudaFree(d_in));
        }
        currElements = blocks;
        d_in = d_out;
        d_out = NULL;
    }
    *d_result = d_in; // now contains the single reduced value
}

// Processes one chunk (here, the entire array) completely in one stream.
static void processOneChunkSingle(const float* h_data, const float* h_weights,
                                  int chunkSize, int threadsPerBlock,
                                  float* h_partialWeighted, float* h_partialWeight)
{
    size_t bytes = chunkSize * sizeof(float);
    float *d_data = NULL, *d_weights = NULL, *d_prod = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_prod, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, bytes, cudaMemcpyHostToDevice));
    int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
    multiplyKernel<<<blocks, threadsPerBlock>>>(d_data, d_weights, d_prod, chunkSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    float *d_chunkWeighted = NULL, *d_chunkWeight = NULL;
    performReductionAsync(d_prod,    chunkSize, threadsPerBlock, 0, &d_chunkWeighted);
    performReductionAsync(d_weights, chunkSize, threadsPerBlock, 0, &d_chunkWeight);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partialWeighted, d_chunkWeighted, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partialWeight,   d_chunkWeight,   sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_chunkWeighted));
    CUDA_CHECK(cudaFree(d_chunkWeight));
}

// pipelineAllSingle: processes the entire data as one chunk, times it, and returns the weighted average.
static float pipelineAllSingle(float* h_data, float* h_weights, int numElements, int threadsPerBlock)
{
    float partialWeighted = 0.0f, partialWeight = 0.0f;
    cudaEvent_t overallStart, overallStop;
    CUDA_CHECK(cudaEventCreate(&overallStart));
    CUDA_CHECK(cudaEventCreate(&overallStop));
    CUDA_CHECK(cudaEventRecord(overallStart, 0));

    processOneChunkSingle(h_data, h_weights, numElements, threadsPerBlock,
                          &partialWeighted, &partialWeight);

    CUDA_CHECK(cudaEventRecord(overallStop, 0));
    CUDA_CHECK(cudaEventSynchronize(overallStop));
    float overallMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&overallMs, overallStart, overallStop));

    float finalAvg = (partialWeight > 0.0f) ? (partialWeighted / partialWeight) : 0.0f;
    printf("Weighted Average = %.4f\n", finalAvg);
    printf("Overall Time (ms) = %.4f\n", overallMs);

    CUDA_CHECK(cudaEventDestroy(overallStart));
    CUDA_CHECK(cudaEventDestroy(overallStop));
    return finalAvg;
}

// Allocates and initializes host arrays with random data.
static void createHostArrays(float** h_data, float** h_weights, int n)
{
    *h_data = (float*)malloc(n * sizeof(float));
    *h_weights = (float*)malloc(n * sizeof(float));
    if (!(*h_data) || !(*h_weights)) {
        fprintf(stderr, "Host malloc failed.\n");
        exit(EXIT_FAILURE);
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        (*h_data)[i] = (float)(rand() % 100);
        (*h_weights)[i] = (float)(rand() % 10 + 1);
    }
}

// ---------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <num_elements> <threads_per_block>\n", argv[0]);
        return 1;
    }

    int numElements = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);

    float* h_data = NULL;
    float* h_weights = NULL;
    createHostArrays(&h_data, &h_weights, numElements);

    float finalAvg = pipelineAllSingle(h_data, h_weights, numElements, threadsPerBlock);
    printf("Done. Weighted average = %.4f\n", finalAvg);

    free(h_data);
    free(h_weights);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
