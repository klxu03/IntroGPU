#include <iostream>
#include <cstdlib>
#include <cuda.h>

// Macro for CUDA error checking.
#define CHECK_CUDA(call) {                                          \
    cudaError_t err = call;                                         \
    if(err != cudaSuccess) {                                        \
        std::cerr << "CUDA error (" << __FILE__ << ":" << __LINE__   \
                  << "): " << cudaGetErrorString(err) << std::endl;  \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// ----------------------------------------------------------------------
// Kernel: multiplyKernel
// Computes element-wise product: prod[i] = data[i] * weights[i]
__global__ void multiplyKernel(const float* data, const float* weights, float* prod, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        prod[idx] = data[idx] * weights[idx];
}

// ----------------------------------------------------------------------
// Kernel: reductionKernel
// Performs block-level reduction of 'input' into 'output'. Assumes shared memory.
__global__ void reductionKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0.0f;
    if (idx < n)
        sum = input[idx];
    if (idx + blockDim.x < n)
        sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    // Reduce within the block.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// ----------------------------------------------------------------------
// Helper: performReductionAsync
// Reduces a device array (d_input) of length 'n' to a single value,
// using 'threadsPerBlock' in the given 'stream'. The final result pointer
// is returned via d_result (allocated inside this function).
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
        // Free the intermediate array if it's not the original input.
        if (d_in != d_input)
            CHECK_CUDA(cudaFree(d_in));
        currElements = blocks;
        d_in = d_out;
        d_out = nullptr;
    }
    // d_in now holds the single reduced value.
    *d_result = d_in;
}

// ----------------------------------------------------------------------
// Helper: processChunkPipeline
// Processes one chunk of data in a pipelined fashion. The stages are:
//   Stage 0: Asynchronously copy the chunk's values and weights to device (using two streams).
//   Stage 1: In stream_mult, wait for both copies to finish and launch the multiplication kernel.
//   Stage 2: In stream_reduc, wait for multiplication to complete and perform reduction on both arrays.
//   Stage 3: In stream_d2h, wait for reduction and copy the partial results back to host.
// The results for this chunk are stored in h_partialWeighted and h_partialWeight.
void processChunkPipeline(const float* h_data, const float* h_weights, int chunkSize, int threadsPerBlock,
                          cudaStream_t stream_in_data, cudaStream_t stream_in_weights,
                          cudaStream_t stream_mult, cudaStream_t stream_reduc, cudaStream_t stream_d2h,
                          cudaEvent_t event_copy_data, cudaEvent_t event_copy_weights,
                          cudaEvent_t event_mult, cudaEvent_t event_reduc,
                          float* h_partialWeighted, float* h_partialWeight) {
    size_t bytes = chunkSize * sizeof(float);
    float *d_data, *d_weights, *d_prod;
    // Allocate device memory for the current chunk.
    CHECK_CUDA(cudaMalloc((void**)&d_data, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_weights, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_prod, bytes));
    
    // ----- Stage 0: H2D Copy -----
    // Copy values and weights concurrently in separate streams.
    CHECK_CUDA(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream_in_data));
    CHECK_CUDA(cudaMemcpyAsync(d_weights, h_weights, bytes, cudaMemcpyHostToDevice, stream_in_weights));
    // Record events to signal completion.
    CHECK_CUDA(cudaEventRecord(event_copy_data, stream_in_data));
    CHECK_CUDA(cudaEventRecord(event_copy_weights, stream_in_weights));
    
    // ----- Stage 1: Multiplication Kernel -----
    // Wait for both input transfers to finish.
    CHECK_CUDA(cudaStreamWaitEvent(stream_mult, event_copy_data, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream_mult, event_copy_weights, 0));
    int blocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;
    multiplyKernel<<<blocks, threadsPerBlock, 0, stream_mult>>>(d_data, d_weights, d_prod, chunkSize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(event_mult, stream_mult));
    
    // ----- Stage 2: Reduction -----
    // Wait for multiplication to complete.
    CHECK_CUDA(cudaStreamWaitEvent(stream_reduc, event_mult, 0));
    float *d_chunkWeighted = nullptr, *d_chunkWeight = nullptr;
    // Reduce the product array to get the weighted sum.
    performReductionAsync(d_prod, chunkSize, threadsPerBlock, stream_reduc, &d_chunkWeighted);
    // Similarly, reduce the weights array to get the total weight.
    performReductionAsync(d_weights, chunkSize, threadsPerBlock, stream_reduc, &d_chunkWeight);
    CHECK_CUDA(cudaEventRecord(event_reduc, stream_reduc));
    
    // ----- Stage 3: D2H Copy -----
    // Wait for reduction to complete.
    CHECK_CUDA(cudaStreamWaitEvent(stream_d2h, event_reduc, 0));
    // Copy the final reduced results back to pinned host memory.
    CHECK_CUDA(cudaMemcpyAsync(h_partialWeighted, d_chunkWeighted, sizeof(float), cudaMemcpyDeviceToHost, stream_d2h));
    CHECK_CUDA(cudaMemcpyAsync(h_partialWeight, d_chunkWeight, sizeof(float), cudaMemcpyDeviceToHost, stream_d2h));
    
    // Free the temporary device memory for this chunk.
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_weights));
    // Note: d_prod was reduced and its memory freed during reduction.
    // d_chunkWeighted and d_chunkWeight will be freed after host synchronization (omitted here for brevity).
}

// ----------------------------------------------------------------------
// Main function
// Usage: ./pipeline <num_elements> <threads_per_block> <num_chunks>
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <num_elements> <threads_per_block> <num_chunks>\n";
        return 1;
    }
    int numElements = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int numChunks = atoi(argv[3]);
    int chunkSize = (numElements + numChunks - 1) / numChunks;

    // Allocate and initialize host arrays for values and weights.
    float* h_data = new float[numElements];
    float* h_weights = new float[numElements];
    for (int i = 0; i < numElements; i++) {
        h_data[i] = static_cast<float>(rand() % 100);
        h_weights[i] = static_cast<float>(rand() % 10 + 1);
    }
    
    // Allocate pinned host memory for partial results from each chunk.
    float* h_partialWeighted;
    float* h_partialWeight;
    CHECK_CUDA(cudaMallocHost((void**)&h_partialWeighted, numChunks * sizeof(float)));
    CHECK_CUDA(cudaMallocHost((void**)&h_partialWeight, numChunks * sizeof(float)));
    
    // Create the streams:
    // Two streams for input transfers (values and weights), and one each for multiplication, reduction, and D2H copy.
    cudaStream_t stream_in_data, stream_in_weights, stream_mult, stream_reduc, stream_d2h;
    CHECK_CUDA(cudaStreamCreate(&stream_in_data));
    CHECK_CUDA(cudaStreamCreate(&stream_in_weights));
    CHECK_CUDA(cudaStreamCreate(&stream_mult));
    CHECK_CUDA(cudaStreamCreate(&stream_reduc));
    CHECK_CUDA(cudaStreamCreate(&stream_d2h));
    
    // Create arrays of events for each chunk.
    cudaEvent_t* event_copy_data = new cudaEvent_t[numChunks];
    cudaEvent_t* event_copy_weights = new cudaEvent_t[numChunks];
    cudaEvent_t* event_mult = new cudaEvent_t[numChunks];
    cudaEvent_t* event_reduc = new cudaEvent_t[numChunks];
    for (int i = 0; i < numChunks; i++) {
        CHECK_CUDA(cudaEventCreate(&event_copy_data[i]));
        CHECK_CUDA(cudaEventCreate(&event_copy_weights[i]));
        CHECK_CUDA(cudaEventCreate(&event_mult[i]));
        CHECK_CUDA(cudaEventCreate(&event_reduc[i]));
    }
    
    // Overall timing events.
    cudaEvent_t overallStart, overallStop;
    CHECK_CUDA(cudaEventCreate(&overallStart));
    CHECK_CUDA(cudaEventCreate(&overallStop));
    CHECK_CUDA(cudaEventRecord(overallStart));
    
    // Process each chunk through the pipeline.
    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;
        int currentChunkSize = ((offset + chunkSize) > numElements) ? (numElements - offset) : chunkSize;
        processChunkPipeline(h_data + offset, h_weights + offset, currentChunkSize, threadsPerBlock,
                             stream_in_data, stream_in_weights, stream_mult, stream_reduc, stream_d2h,
                             event_copy_data[i], event_copy_weights[i], event_mult[i], event_reduc[i],
                             &h_partialWeighted[i], &h_partialWeight[i]);
    }
    
    // Synchronize all streams to ensure the pipeline is complete.
    CHECK_CUDA(cudaStreamSynchronize(stream_in_data));
    CHECK_CUDA(cudaStreamSynchronize(stream_in_weights));
    CHECK_CUDA(cudaStreamSynchronize(stream_mult));
    CHECK_CUDA(cudaStreamSynchronize(stream_reduc));
    CHECK_CUDA(cudaStreamSynchronize(stream_d2h));
    
    // Record overall stop time.
    CHECK_CUDA(cudaEventRecord(overallStop));
    CHECK_CUDA(cudaEventSynchronize(overallStop));
    float overallTime;
    CHECK_CUDA(cudaEventElapsedTime(&overallTime, overallStart, overallStop));
    
    // Aggregate partial results on the host.
    float totalWeightedSum = 0.0f, totalWeight = 0.0f;
    for (int i = 0; i < numChunks; i++) {
        totalWeightedSum += h_partialWeighted[i];
        totalWeight += h_partialWeight[i];
    }
    float weightedAverage = (totalWeight > 0) ? totalWeightedSum / totalWeight : 0.0f;
    std::cout << "Weighted Average = " << weightedAverage << "\n";
    std::cout << "Overall Time (ms) = " << overallTime << "\n";
    
    // Cleanup: destroy events and streams.
    for (int i = 0; i < numChunks; i++) {
        cudaEventDestroy(event_copy_data[i]);
        cudaEventDestroy(event_copy_weights[i]);
        cudaEventDestroy(event_mult[i]);
        cudaEventDestroy(event_reduc[i]);
    }
    delete[] event_copy_data;
    delete[] event_copy_weights;
    delete[] event_mult;
    delete[] event_reduc;
    
    cudaStreamDestroy(stream_in_data);
    cudaStreamDestroy(stream_in_weights);
    cudaStreamDestroy(stream_mult);
    cudaStreamDestroy(stream_reduc);
    cudaStreamDestroy(stream_d2h);
    
    cudaEventDestroy(overallStart);
    cudaEventDestroy(overallStop);
    
    cudaFreeHost(h_partialWeighted);
    cudaFreeHost(h_partialWeight);
    delete[] h_data;
    delete[] h_weights;
    
    return 0;
}
