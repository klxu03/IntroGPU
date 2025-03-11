#include <iostream>
#include <cstdlib>
#include <cuda.h>

#define CHECK_CUDA(call) {                                          \
    cudaError_t err = call;                                         \
    if(err != cudaSuccess) {                                        \
        std::cerr << "CUDA error (" << __FILE__ << ":" << __LINE__   \
                  << "): " << cudaGetErrorString(err) << std::endl;  \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// ----------------------------------------------------------------------
// multiplyKernel: computes element-wise product: prod[i] = data[i] * weights[i]
__global__ void multiplyKernel(const float* data, const float* weights, float* prod, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        prod[idx] = data[idx] * weights[idx];
    }
}

// ----------------------------------------------------------------------
// reductionKernel: sums an array using shared memory, writes block sums to 'output'
__global__ void reductionKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
    }
    if (idx + blockDim.x < n) {
        sum += input[idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ----------------------------------------------------------------------
// performReductionAsync (same as before, but we'll use it in a single stream)
void performReductionAsync(float* d_input, int n, int threadsPerBlock, cudaStream_t stream, float** d_result) {
    int currElements = n;
    float* d_in = d_input;
    float* d_out = nullptr;

    while (currElements > 1) {
        int blocks = (currElements + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
        CHECK_CUDA(cudaMalloc((void**)&d_out, blocks * sizeof(float)));
        size_t sharedMem = threadsPerBlock * sizeof(float);

        reductionKernel<<<blocks, threadsPerBlock, sharedMem, stream>>>(d_in, d_out, currElements);
        CHECK_CUDA(cudaGetLastError());

        // Free the old array if it's not the original input.
        if (d_in != d_input) {
            CHECK_CUDA(cudaFree(d_in));
        }
        currElements = blocks;
        d_in = d_out;
        d_out = nullptr;
    }
    // d_in now has the single reduced value
    *d_result = d_in;
}

// ----------------------------------------------------------------------
// processChunkSingle: processes one chunk entirely in a single stream
// 1) Copy data & weights to device (synchronously or async + sync)
// 2) multiplyKernel
// 3) reduction on prod and on weights
// 4) copy final partial results back
void processChunkSingle(const float* h_data, const float* h_weights, int chunkSize, int threadsPerBlock,
                        float* h_partialWeighted, float* h_partialWeight) {
    cudaStream_t stream; 
    CHECK_CUDA(cudaStreamCreate(&stream));  // single stream for everything in this chunk

    size_t bytes = chunkSize * sizeof(float);
    float *d_data, *d_weights, *d_prod;
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_weights, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_prod, bytes));

    // Copy data from host to device (blocking or synchronous approach)
    CHECK_CUDA(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, bytes, cudaMemcpyHostToDevice));

    // Launch multiplyKernel in the same stream
    int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
    multiplyKernel<<<blocks, threadsPerBlock, 0, stream>>>(d_data, d_weights, d_prod, chunkSize);
    CHECK_CUDA(cudaGetLastError());

    // We can either synchronize or let the next operation wait on the same stream
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Perform reduction on d_prod -> partial weighted sum
    float* d_chunkWeighted = nullptr;
    performReductionAsync(d_prod, chunkSize, threadsPerBlock, stream, &d_chunkWeighted);

    // Perform reduction on d_weights -> partial weight sum
    float* d_chunkWeight = nullptr;
    performReductionAsync(d_weights, chunkSize, threadsPerBlock, stream, &d_chunkWeight);

    // Synchronize to ensure reductions complete
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy the single reduced results back to host
    CHECK_CUDA(cudaMemcpy(h_partialWeighted, d_chunkWeighted, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_partialWeight, d_chunkWeight, sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_weights));
    if (d_chunkWeighted) CHECK_CUDA(cudaFree(d_chunkWeighted));
    if (d_chunkWeight) CHECK_CUDA(cudaFree(d_chunkWeight));
    // d_prod was freed in the first step of the reduction, if not, free it:
    // But let's be safe:
    // (If chunkSize == 1, we never called free on d_prod)
    CHECK_CUDA(cudaFree(d_prod));

    CHECK_CUDA(cudaStreamDestroy(stream));
}

// ----------------------------------------------------------------------
// Main function for single-stream, sequential chunk processing
// Usage: ./pipeline_single <num_elements> <threads_per_block> <num_chunks>
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <num_elements> <threads_per_block> <num_chunks>\n";
        return 1;
    }
    int numElements = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int numChunks = atoi(argv[3]);
    int chunkSize = (numElements + numChunks - 1) / numChunks;

    // Allocate and initialize host arrays
    float* h_data = new float[numElements];
    float* h_weights = new float[numElements];
    for (int i = 0; i < numElements; i++) {
        h_data[i] = static_cast<float>(rand() % 100);
        h_weights[i] = static_cast<float>(rand() % 10 + 1);
    }

    // Allocate host memory for partial results
    float* h_partialWeighted = new float[numChunks];
    float* h_partialWeight = new float[numChunks];

    // Overall timing
    cudaEvent_t overallStart, overallStop;
    CHECK_CUDA(cudaEventCreate(&overallStart));
    CHECK_CUDA(cudaEventCreate(&overallStop));
    CHECK_CUDA(cudaEventRecord(overallStart));

    // Process each chunk in a single stream, one after another
    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;
        int currentChunkSize = (offset + chunkSize > numElements) ? (numElements - offset) : chunkSize;

        processChunkSingle(h_data + offset, h_weights + offset, currentChunkSize,
                           threadsPerBlock, &h_partialWeighted[i], &h_partialWeight[i]);
    }

    // Record end time
    CHECK_CUDA(cudaEventRecord(overallStop));
    CHECK_CUDA(cudaEventSynchronize(overallStop));
    float overallTime = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&overallTime, overallStart, overallStop));

    // Compute final weighted average on host
    float totalWeightedSum = 0.0f, totalWeight = 0.0f;
    for (int i = 0; i < numChunks; i++) {
        totalWeightedSum += h_partialWeighted[i];
        totalWeight += h_partialWeight[i];
    }
    float weightedAverage = (totalWeight > 0) ? (totalWeightedSum / totalWeight) : 0.0f;

    std::cout << "Single-Stream Weighted Average = " << weightedAverage << "\n";
    std::cout << "Single-Stream Overall Time (ms) = " << overallTime << "\n";

    // Cleanup
    delete[] h_data;
    delete[] h_weights;
    delete[] h_partialWeighted;
    delete[] h_partialWeight;
    CHECK_CUDA(cudaEventDestroy(overallStart));
    CHECK_CUDA(cudaEventDestroy(overallStop));

    return 0;
}
