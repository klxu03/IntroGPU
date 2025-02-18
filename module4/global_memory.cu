/******************************************************************************
 * global_memory_experiments.cu
 *
 * Demonstrates interleaved vs. non-interleaved array access on CPU and GPU
 * for various array sizes and iteration counts, printing out timing results
 * in CSV format. Also includes a simple bitreverse kernel test with multiple
 * block sizes for more experiments.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

// -----------------------------------------------------------------------------
// Error-checking macro
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(err);                                                 \
    }                                                              \
} while(0)

// -----------------------------------------------------------------------------
// Timing helpers
// -----------------------------------------------------------------------------
static inline cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time, 0);
    cudaEventSynchronize(time);
    return time;
}

static inline float elapsed_time(cudaEvent_t start, cudaEvent_t end)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
}

// -----------------------------------------------------------------------------
// Interleaved vs. Non-interleaved Data Structures
// -----------------------------------------------------------------------------
typedef struct {
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
} INTERLEAVED_T;

// Non-interleaved: 4 separate arrays
typedef struct {
    unsigned int *a;
    unsigned int *b;
    unsigned int *c;
    unsigned int *d;
} NON_INTERLEAVED_T;

// -----------------------------------------------------------------------------
// CPU addition: Interleaved
// -----------------------------------------------------------------------------
float cpu_add_interleaved(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                          unsigned int num_elements, unsigned int iter)
{
    cudaEvent_t start = get_time();
    for (unsigned int i = 0; i < num_elements; i++) {
        for (unsigned int j = 0; j < iter; j++) {
            dest[i].a += src[i].a;
            dest[i].b += src[i].b;
            dest[i].c += src[i].c;
            dest[i].d += src[i].d;
        }
    }
    cudaEvent_t end = get_time();
    return elapsed_time(start, end);
}

// -----------------------------------------------------------------------------
// CPU addition: Non-interleaved
// -----------------------------------------------------------------------------
float cpu_add_noninterleaved(NON_INTERLEAVED_T &dest, const NON_INTERLEAVED_T &src,
                             unsigned int num_elements, unsigned int iter)
{
    cudaEvent_t start = get_time();
    for (unsigned int i = 0; i < num_elements; i++) {
        for (unsigned int j = 0; j < iter; j++) {
            dest.a[i] += src.a[i];
            dest.b[i] += src.b[i];
            dest.c[i] += src.c[i];
            dest.d[i] += src.d[i];
        }
    }
    cudaEvent_t end = get_time();
    return elapsed_time(start, end);
}

// -----------------------------------------------------------------------------
// GPU kernel: Interleaved
// -----------------------------------------------------------------------------
__global__ void add_kernel_interleaved(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                                       unsigned int num_elements, unsigned int iter)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        for (unsigned int i = 0; i < iter; i++) {
            dest[tid].a += src[tid].a;
            dest[tid].b += src[tid].b;
            dest[tid].c += src[tid].c;
            dest[tid].d += src[tid].d;
        }
    }
}

// -----------------------------------------------------------------------------
// GPU kernel: Non-interleaved
// -----------------------------------------------------------------------------
__global__ void add_kernel_noninterleaved(NON_INTERLEAVED_T dest, NON_INTERLEAVED_T src,
                                          unsigned int num_elements, unsigned int iter)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        for (unsigned int i = 0; i < iter; i++) {
            dest.a[tid] += src.a[tid];
            dest.b[tid] += src.b[tid];
            dest.c[tid] += src.c[tid];
            dest.d[tid] += src.d[tid];
        }
    }
}

// -----------------------------------------------------------------------------
// GPU addition: Interleaved
// -----------------------------------------------------------------------------
float gpu_add_interleaved(INTERLEAVED_T *h_dest, const INTERLEAVED_T *h_src,
                          unsigned int num_elements, unsigned int iter)
{
    // Allocate device memory
    size_t bytes = num_elements * sizeof(INTERLEAVED_T);
    INTERLEAVED_T *d_dest = nullptr, *d_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dest, bytes));
    CUDA_CHECK(cudaMalloc(&d_src,  bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dest, 0, bytes)); // start from zero to see the addition cost

    // Launch kernel
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    cudaEvent_t start = get_time();
    add_kernel_interleaved<<<grid, block>>>(d_dest, d_src, num_elements, iter);
    cudaEvent_t end = get_time();
    CUDA_CHECK(cudaDeviceSynchronize()); // ensure kernel finishes

    float ms = elapsed_time(start, end);

    // Copy results back (optional if you just want the timing)
    CUDA_CHECK(cudaMemcpy(h_dest, d_dest, bytes, cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaFree(d_dest));
    CUDA_CHECK(cudaFree(d_src));
    return ms;
}

// -----------------------------------------------------------------------------
// GPU addition: Non-interleaved
// -----------------------------------------------------------------------------
float gpu_add_noninterleaved(NON_INTERLEAVED_T &h_dest, const NON_INTERLEAVED_T &h_src,
                             unsigned int num_elements, unsigned int iter)
{
    // Allocate device memory
    size_t bytes = num_elements * sizeof(unsigned int);
    NON_INTERLEAVED_T d_dest, d_src;
    CUDA_CHECK(cudaMalloc(&d_dest.a, bytes));
    CUDA_CHECK(cudaMalloc(&d_dest.b, bytes));
    CUDA_CHECK(cudaMalloc(&d_dest.c, bytes));
    CUDA_CHECK(cudaMalloc(&d_dest.d, bytes));

    CUDA_CHECK(cudaMalloc(&d_src.a, bytes));
    CUDA_CHECK(cudaMalloc(&d_src.b, bytes));
    CUDA_CHECK(cudaMalloc(&d_src.c, bytes));
    CUDA_CHECK(cudaMalloc(&d_src.d, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_src.a, h_src.a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src.b, h_src.b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src.c, h_src.c, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src.d, h_src.d, bytes, cudaMemcpyHostToDevice));

    // Zero out dest
    CUDA_CHECK(cudaMemset(d_dest.a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dest.b, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dest.c, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dest.d, 0, bytes));

    // Launch kernel
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    cudaEvent_t start = get_time();
    add_kernel_noninterleaved<<<grid, block>>>(d_dest, d_src, num_elements, iter);
    cudaEvent_t end = get_time();
    CUDA_CHECK(cudaDeviceSynchronize());

    float ms = elapsed_time(start, end);

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_dest.a, d_dest.a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.b, d_dest.b, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.c, d_dest.c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.d, d_dest.d, bytes, cudaMemcpyDeviceToHost));

    // Clean up
    CUDA_CHECK(cudaFree(d_dest.a));
    CUDA_CHECK(cudaFree(d_dest.b));
    CUDA_CHECK(cudaFree(d_dest.c));
    CUDA_CHECK(cudaFree(d_dest.d));
    CUDA_CHECK(cudaFree(d_src.a));
    CUDA_CHECK(cudaFree(d_src.b));
    CUDA_CHECK(cudaFree(d_src.c));
    CUDA_CHECK(cudaFree(d_src.d));

    return ms;
}

// -----------------------------------------------------------------------------
// Simple bitreverse kernel for extra experiments
// -----------------------------------------------------------------------------
__host__ __device__ unsigned int bitreverse_func(unsigned int number) {
    number = ((0xf0f0f0f0 & number) >> 4)  | ((0x0f0f0f0f & number) << 4);
    number = ((0xcccccccc & number) >> 2)  | ((0x33333333 & number) << 2);
    number = ((0xaaaaaaaa & number) >> 1)  | ((0x55555555 & number) << 1);
    return number;
}

__global__ void bitreverse_kernel(unsigned int *data, unsigned int size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = bitreverse_func(data[tid]);
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    printf("=== Interleaved vs Non-Interleaved Memory Experiments ===\n");
    printf("CSV Format: size,iter,cpu_interleaved_ms,gpu_interleaved_ms,cpu_noninterleaved_ms,gpu_noninterleaved_ms\n");

    // We'll test array sizes from 256 to 131072, doubling each time.
    // We'll test iteration counts from 1, 2, 4, 8, 16, 32.
    unsigned int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    unsigned int iters[] = {1, 2, 4, 8, 16, 32};

    for (unsigned int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        unsigned int num_elements = sizes[s];

        // Allocate host memory for interleaved
        INTERLEAVED_T *host_src_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        INTERLEAVED_T *host_dest_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));

        // Allocate host memory for non-interleaved
        NON_INTERLEAVED_T host_src_n, host_dest_n;
        host_src_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));

        host_dest_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));

        // Initialize data
        for (unsigned int i = 0; i < num_elements; i++) {
            host_src_i[i].a = i;
            host_src_i[i].b = i + 1;
            host_src_i[i].c = i + 2;
            host_src_i[i].d = i + 3;

            // mirror data for non-interleaved
            host_src_n.a[i] = i;
            host_src_n.b[i] = i + 1;
            host_src_n.c[i] = i + 2;
            host_src_n.d[i] = i + 3;

            // zero out dest
            host_dest_i[i].a = 0;
            host_dest_i[i].b = 0;
            host_dest_i[i].c = 0;
            host_dest_i[i].d = 0;

            host_dest_n.a[i] = 0;
            host_dest_n.b[i] = 0;
            host_dest_n.c[i] = 0;
            host_dest_n.d[i] = 0;
        }

        // Now loop over iteration counts
        for (unsigned int it = 0; it < sizeof(iters)/sizeof(iters[0]); it++) {
            unsigned int iter = iters[it];

            // CPU times
            float cpu_int_ms = cpu_add_interleaved(host_dest_i, host_src_i, num_elements, iter);

            // Re-zero out host_dest_i for a fair GPU test
            for (unsigned int i = 0; i < num_elements; i++) {
                host_dest_i[i].a = 0;
                host_dest_i[i].b = 0;
                host_dest_i[i].c = 0;
                host_dest_i[i].d = 0;
            }

            float gpu_int_ms = gpu_add_interleaved(host_dest_i, host_src_i, num_elements, iter);

            // CPU times (non-interleaved)
            float cpu_nint_ms = cpu_add_noninterleaved(host_dest_n, host_src_n, num_elements, iter);

            // Re-zero out host_dest_n for GPU
            for (unsigned int i = 0; i < num_elements; i++) {
                host_dest_n.a[i] = 0;
                host_dest_n.b[i] = 0;
                host_dest_n.c[i] = 0;
                host_dest_n.d[i] = 0;
            }

            float gpu_nint_ms = gpu_add_noninterleaved(host_dest_n, host_src_n, num_elements, iter);

            // Print CSV line
            printf("%u,%u,%.4f,%.4f,%.4f,%.4f\n",
                   num_elements, iter, cpu_int_ms, gpu_int_ms, cpu_nint_ms, gpu_nint_ms);
        }

        // Free memory
        free(host_src_i);
        free(host_dest_i);

        free(host_src_n.a);  free(host_src_n.b);
        free(host_src_n.c);  free(host_src_n.d);
        free(host_dest_n.a); free(host_dest_n.b);
        free(host_dest_n.c); free(host_dest_n.d);
    }

    // -------------------------------------------------------------------------
    // Extra experiment: bitreverse with different block sizes
    // -------------------------------------------------------------------------
    printf("\n=== Bitreverse Experiments ===\n");
    printf("CSV Format: arraySize,blockSize,bitreverseTime(ms)\n");

    // Let’s pick an array size for bitreverse
    const unsigned int BITREVERSE_SIZE = 1 << 16; // 65536
    unsigned int *h_data = (unsigned int*) malloc(BITREVERSE_SIZE * sizeof(unsigned int));
    for (unsigned int i = 0; i < BITREVERSE_SIZE; i++) {
        h_data[i] = i;
    }

    // Copy to device once, then do multiple block sizes
    unsigned int *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, BITREVERSE_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, BITREVERSE_SIZE * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Test block sizes from 64 to 1024
    for (unsigned int blockSize = 64; blockSize <= 1024; blockSize *= 2) {
        dim3 block(blockSize);
        dim3 grid((BITREVERSE_SIZE + blockSize - 1) / blockSize);

        // Re-initialize device data each time so timing is consistent
        CUDA_CHECK(cudaMemcpy(d_data, h_data, BITREVERSE_SIZE * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        cudaEvent_t start = get_time();
        bitreverse_kernel<<<grid, block>>>(d_data, BITREVERSE_SIZE);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t end = get_time();

        float ms = elapsed_time(start, end);
        printf("%u,%u,%.4f\n", BITREVERSE_SIZE, blockSize, ms);
    }

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}