/****************************************************************************
 * File: streams.cu
 *
 * Cuda streams pipeline weighted average
 *
 * Usage:
 *   nvcc streams.cu -o streams
 *   ./streams <num_elements> <threads_per_block> <num_chunks>
 *
 * Example:
 *   ./streams 1000000 256 8
 *
 * This program splits an array of 'num_elements' into 'num_chunks', then for each chunk:
 *   1) Copies data & weights asynchronously to the GPU (two streams)
 *   2) Multiplies them (another stream)
 *   3) Reduces both arrays to a single sum each (another stream)
 *   4) Copies those partial sums back (final stream)
 * Finally, it aggregates the partial sums into a final weighted average on the host and prints total timing in milliseconds.
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
__global__ void multiplyKernel(const float* d_data, const float* d_weights, float* d_prod, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_prod[idx] = d_data[idx] * d_weights[idx];
    }
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

// processOneChunk: runs the 4 pipeline stages on a single chunk.
static void processOneChunk(const float* h_data, const float* h_weights,
                            int chunkSize, int threadsPerBlock,
                            cudaStream_t s_in_data, cudaStream_t s_in_weights,
                            cudaStream_t s_mult, cudaStream_t s_reduc, cudaStream_t s_d2h,
                            cudaEvent_t e_copy_data, cudaEvent_t e_copy_weights,
                            cudaEvent_t e_mult, cudaEvent_t e_reduc,
                            float* h_partialWeighted, float* h_partialWeight)
{
    size_t bytes = chunkSize * sizeof(float);
    float *d_data = NULL, *d_weights = NULL;
    // will store the product of the data and weights
    float *d_prod = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_weights, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_prod, bytes));

    // Stage 0: Asynchronously copy data & weights in separate streams
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, s_in_data));
    CUDA_CHECK(cudaMemcpyAsync(d_weights, h_weights, bytes, cudaMemcpyHostToDevice, s_in_weights));
    CUDA_CHECK(cudaEventRecord(e_copy_data, s_in_data));
    CUDA_CHECK(cudaEventRecord(e_copy_weights, s_in_weights));

    // Stage 1: multiplyKernel waits on data/weights
    CUDA_CHECK(cudaStreamWaitEvent(s_mult, e_copy_data, 0));
    CUDA_CHECK(cudaStreamWaitEvent(s_mult, e_copy_weights, 0));
    int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
    multiplyKernel<<<blocks, threadsPerBlock, 0, s_mult>>>(d_data, d_weights, d_prod, chunkSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e_mult, s_mult));

    // Stage 2: reduction waits on multiply
    CUDA_CHECK(cudaStreamWaitEvent(s_reduc, e_mult, 0));
    float *d_chunkWeighted = NULL, *d_chunkWeight = NULL;
    performReductionAsync(d_prod, chunkSize, threadsPerBlock, s_reduc, &d_chunkWeighted); // sum of products
    performReductionAsync(d_weights, chunkSize, threadsPerBlock, s_reduc, &d_chunkWeight); // sum of weights
    CUDA_CHECK(cudaEventRecord(e_reduc, s_reduc));

    // Stage 3: copy partial results back waits on reduction
    CUDA_CHECK(cudaStreamWaitEvent(s_d2h, e_reduc, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_partialWeighted, d_chunkWeighted, sizeof(float),
                               cudaMemcpyDeviceToHost, s_d2h));
    CUDA_CHECK(cudaMemcpyAsync(h_partialWeight, d_chunkWeight, sizeof(float),
                               cudaMemcpyDeviceToHost, s_d2h));

    // Freed once host is done using them
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_weights));
    // d_prod was freed inside performReductionAsync
    CUDA_CHECK(cudaFree(d_chunkWeighted));
    CUDA_CHECK(cudaFree(d_chunkWeight));
}

// pipelineChunks: sets up streams, events, processes each chunk, and aggregates.
static float pipelineChunks(float* h_data, float* h_weights, int numElements,
                            int threadsPerBlock, int numChunks)
{
    int chunkSize = (numElements + numChunks - 1) / numChunks;

    // Create streams
    cudaStream_t s_in_data, s_in_weights, s_mult, s_reduc, s_d2h;
    CUDA_CHECK(cudaStreamCreate(&s_in_data)); // input data
    CUDA_CHECK(cudaStreamCreate(&s_in_weights)); // input weights
    CUDA_CHECK(cudaStreamCreate(&s_mult)); // stream for vector multiplication
    CUDA_CHECK(cudaStreamCreate(&s_reduc)); // stream for reduction
    CUDA_CHECK(cudaStreamCreate(&s_d2h)); // stream for copying back to host

    // Create events (one set per chunk)
    cudaEvent_t *e_copy_data = new cudaEvent_t[numChunks];
    cudaEvent_t *e_copy_weights = new cudaEvent_t[numChunks];
    cudaEvent_t *e_mult = new cudaEvent_t[numChunks];
    cudaEvent_t *e_reduc = new cudaEvent_t[numChunks];
    for (int i = 0; i < numChunks; i++) {
        CUDA_CHECK(cudaEventCreate(&e_copy_data[i]));
        CUDA_CHECK(cudaEventCreate(&e_copy_weights[i]));
        CUDA_CHECK(cudaEventCreate(&e_mult[i]));
        CUDA_CHECK(cudaEventCreate(&e_reduc[i]));
    }

    // Allocate pinned host memory for partial results
    // partialWeighted: sum of products after they've been weighted in a chunk
    // partialWeight: sum of weights in a chunk
    float *h_partialWeighted = NULL, *h_partialWeight = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&h_partialWeighted, numChunks * sizeof(float)));
    CUDA_CHECK(cudaMallocHost((void**)&h_partialWeight, numChunks * sizeof(float)));

    // Overall timing
    cudaEvent_t overallStart, overallStop;
    CUDA_CHECK(cudaEventCreate(&overallStart));
    CUDA_CHECK(cudaEventCreate(&overallStop));
    CUDA_CHECK(cudaEventRecord(overallStart, 0));

    // Launch pipeline for each chunk
    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;
        int size = (offset + chunkSize > numElements) ? (numElements - offset) : chunkSize; // size of the chunk to process
        processOneChunk(h_data + offset, h_weights + offset,
                        size, threadsPerBlock,
                        s_in_data, s_in_weights, s_mult, s_reduc, s_d2h,
                        e_copy_data[i], e_copy_weights[i], e_mult[i], e_reduc[i],
                        &h_partialWeighted[i], &h_partialWeight[i]);
    }

    // Synchronize all streams to ensure pipeline completion
    CUDA_CHECK(cudaStreamSynchronize(s_in_data));
    CUDA_CHECK(cudaStreamSynchronize(s_in_weights));
    CUDA_CHECK(cudaStreamSynchronize(s_mult));
    CUDA_CHECK(cudaStreamSynchronize(s_reduc));
    CUDA_CHECK(cudaStreamSynchronize(s_d2h));

    // Stop timing
    CUDA_CHECK(cudaEventRecord(overallStop, 0));
    CUDA_CHECK(cudaEventSynchronize(overallStop));
    float overallMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&overallMs, overallStart, overallStop));

    // Aggregate partial results on host
    float totalWeighted = 0.0f;
    float totalWeight = 0.0f;
    for (int i = 0; i < numChunks; i++) {
        totalWeighted += h_partialWeighted[i];
        totalWeight += h_partialWeight[i];
    }
    if (totalWeight <= 0.0f) {
        printf("No valid weights found in the input.\n");
        return 0.0f;
    }
    float finalAvg = totalWeighted / totalWeight;

    // Print results
    printf("Weighted Average = %.4f\n", finalAvg);
    printf("Overall Time (ms) = %.4f\n", overallMs);

    // Cleanup
    for (int i = 0; i < numChunks; i++) {
        CUDA_CHECK(cudaEventDestroy(e_copy_data[i]));
        CUDA_CHECK(cudaEventDestroy(e_copy_weights[i]));
        CUDA_CHECK(cudaEventDestroy(e_mult[i]));
        CUDA_CHECK(cudaEventDestroy(e_reduc[i]));
    }
    delete[] e_copy_data;
    delete[] e_copy_weights;
    delete[] e_mult;
    delete[] e_reduc;

    CUDA_CHECK(cudaStreamDestroy(s_in_data));
    CUDA_CHECK(cudaStreamDestroy(s_in_weights));
    CUDA_CHECK(cudaStreamDestroy(s_mult));
    CUDA_CHECK(cudaStreamDestroy(s_reduc));
    CUDA_CHECK(cudaStreamDestroy(s_d2h));

    CUDA_CHECK(cudaFreeHost(h_partialWeighted));
    CUDA_CHECK(cudaFreeHost(h_partialWeight));
    CUDA_CHECK(cudaEventDestroy(overallStart));
    CUDA_CHECK(cudaEventDestroy(overallStop));

    return finalAvg;
}

// createHostArrays: allocate host arrays and fill with random data
static void createHostArrays(float** h_data, float** h_weights, int n)
{
    // Using regular malloc for host memory (pinned for partial results only)
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
    int numElements     = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int numChunks       = atoi(argv[3]);

    float* h_data = NULL;
    float* h_weights = NULL;
    createHostArrays(&h_data, &h_weights, numElements);

    // Run pipeline
    float finalAvg = pipelineChunks(h_data, h_weights, numElements,
                                    threadsPerBlock, numChunks);

    // Print final average (already printed timing inside pipeline)
    printf("Done. Weighted average = %.4f\n", finalAvg);

    free(h_data);
    free(h_weights);
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}