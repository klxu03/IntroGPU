/****************************************************************************
 * File: single.cu
 *
 * Demonstrates a single-stream (non-pipelined) CUDA program that processes 
 * data in chunks. For each chunk, it:
 *   1) Copies data & weights from host to device.
 *   2) Multiplies them.
 *   3) Reduces both arrays to a single sum each.
 *   4) Copies those partial sums back.
 * Finally, it aggregates the partial sums into a final weighted average and
 * prints total timing in milliseconds.
 *
 * Usage:
 *   nvcc single.cu -o single
 *   ./single <num_elements> <threads_per_block> <num_chunks>
 *
 * Example:
 *   ./single 1000000 256 8
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

    // sum the pair of elements thread is responsible for
    float sum = 0.0f;
    if (idx < n) {
        sum = d_in[idx];
    }
    if (idx + blockDim.x < n) {
        sum += d_in[idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    // keep reducing partial sums in half until only one float remains
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
// this performs the full reduction result and stores it in d_result
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

        // Free the old buffer if not the original
        if (d_in != d_input) {
            CUDA_CHECK(cudaFree(d_in));
        }
        currElements = blocks; // old number of blocks is now the number of elements we need to perform reduction on next
        d_in = d_out;
        d_out = NULL;
    }
    *d_result = d_in; // Single reduced value on device
}

// Processes one chunk completely without concurrent streams 
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
    performReductionAsync(d_prod, chunkSize, threadsPerBlock, 0, &d_chunkWeighted);
    performReductionAsync(d_weights, chunkSize, threadsPerBlock, 0, &d_chunkWeight);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_partialWeighted, d_chunkWeighted, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partialWeight, d_chunkWeight,   sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free the device memory
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_weights));
    CUDA_CHECK(cudaFree(d_chunkWeighted));
    CUDA_CHECK(cudaFree(d_chunkWeight));
}

// Sets up processing for all chunks sequentially and aggregates the result.
static float pipelineChunksSingle(float* h_data, float* h_weights, int numElements,
                                  int threadsPerBlock, int numChunks)
{
    int chunkSize = (numElements + numChunks - 1) / numChunks;
    float *h_partialWeighted = NULL, *h_partialWeight = NULL;
    
    CUDA_CHECK(cudaMallocHost((void**)&h_partialWeighted, numChunks * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_partialWeight, numChunks * sizeof(float)));

    cudaEvent_t overallStart, overallStop;
    CUDA_CHECK(cudaEventCreate(&overallStart));
    CUDA_CHECK(cudaEventCreate(&overallStop));
    CUDA_CHECK(cudaEventRecord(overallStart, 0));

    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;
        int size = (offset + chunkSize > numElements) ? (numElements - offset) : chunkSize;
        processOneChunkSingle(h_data + offset, h_weights + offset, size, threadsPerBlock,
                              &h_partialWeighted[i], &h_partialWeight[i]);
    }

    CUDA_CHECK(cudaEventRecord(overallStop, 0));
    CUDA_CHECK(cudaEventSynchronize(overallStop));

    float overallMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&overallMs, overallStart, overallStop));

    float totalWeighted = 0.0f, totalWeight = 0.0f;
    for (int i = 0; i < numChunks; i++) {
        totalWeighted += h_partialWeighted[i];
        totalWeight += h_partialWeight[i];
    }
    if (totalWeight <= 0.0f) {
        printf("No valid weights found in the input.\n");
        return 0.0f;
    }
    float finalAvg = totalWeighted / totalWeight;

    printf("Weighted Average = %.4f\n", finalAvg);
    printf("Overall Time (ms) = %.4f\n", overallMs);
    CUDA_CHECK(cudaFreeHost(h_partialWeighted));
    CUDA_CHECK(cudaFreeHost(h_partialWeight));
    CUDA_CHECK(cudaEventDestroy(overallStart));
    CUDA_CHECK(cudaEventDestroy(overallStop));
    return finalAvg;
}

// Allocates and initializes host arrays with random data.
static void createHostArrays(float** h_data, float** h_weights, int n)
{
    *h_data    = (float*)malloc(n * sizeof(float));
    *h_weights = (float*)malloc(n * sizeof(float));
    if (!(*h_data) || !(*h_weights)) {
        fprintf(stderr, "Host malloc failed.\n");
        exit(EXIT_FAILURE);
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        (*h_data)[i]    = (float)(rand() % 100);
        (*h_weights)[i] = (float)(rand() % 10 + 1);
    }
}

// ---------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <num_elements> <threads_per_block> <num_chunks>\n", argv[0]);
        return 1;
    }
    int numElements = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int numChunks = atoi(argv[3]);

    float* h_data = NULL;
    float* h_weights = NULL;
    createHostArrays(&h_data, &h_weights, numElements);

    float finalAvg = pipelineChunksSingle(h_data, h_weights, numElements, threadsPerBlock, numChunks);

    printf("Done. Weighted average = %.4f\n", finalAvg);

    free(h_data);
    free(h_weights);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}