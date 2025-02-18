/******************************************************************************
 * global_memory.cu
 *
 * Based on original NVIDIA sample code (Copyright 1993-2012 NVIDIA Corporation)
 * and prior experiments, this file now includes additional timing experiments
 * for interleaved versus non-interleaved memory accesses on CPU and GPU,
 * as well as extra bitreverse tests using various block sizes.
 *
 * CSV outputs are provided for each experiment.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error-checking macro
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if(err != cudaSuccess){                                        \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",         \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(err);                                                 \
    }                                                              \
} while(0)

// -----------------------------------------------------------------------------
// Timing helpers using CUDA events
// -----------------------------------------------------------------------------
static inline cudaEvent_t get_time(void) {
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreate(&ev));
    CUDA_CHECK(cudaEventRecord(ev, 0));
    CUDA_CHECK(cudaEventSynchronize(ev));
    return ev;
}

static inline float elapsed_time(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
}

// -----------------------------------------------------------------------------
// Data Structures (as in the original code)
// -----------------------------------------------------------------------------
typedef struct {
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
} INTERLEAVED_T;

typedef struct {
    unsigned int *a;
    unsigned int *b;
    unsigned int *c;
    unsigned int *d;
} NON_INTERLEAVED_T;

// -----------------------------------------------------------------------------
// CPU addition: Interleaved
// -----------------------------------------------------------------------------
float add_test_interleaved_cpu(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                               unsigned int iter, unsigned int num_elements) {
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
float add_test_noninterleaved_cpu(NON_INTERLEAVED_T dest, NON_INTERLEAVED_T src,
                                  unsigned int iter, unsigned int num_elements) {
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
// GPU kernel: Interleaved addition
// -----------------------------------------------------------------------------
__global__ void add_kernel_interleaved(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                                       unsigned int iter, unsigned int num_elements) {
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
// GPU kernel: Non-interleaved addition
// -----------------------------------------------------------------------------
__global__ void add_kernel_noninterleaved(NON_INTERLEAVED_T dest, NON_INTERLEAVED_T src,
                                          unsigned int iter, unsigned int num_elements) {
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
// GPU addition tests
// -----------------------------------------------------------------------------
float add_test_interleaved_gpu(INTERLEAVED_T *h_dest, const INTERLEAVED_T *h_src,
                               unsigned int iter, unsigned int num_elements) {
    size_t bytes = num_elements * sizeof(INTERLEAVED_T);
    INTERLEAVED_T *d_dest, *d_src;
    CUDA_CHECK(cudaMalloc(&d_dest, bytes));
    CUDA_CHECK(cudaMalloc(&d_src,  bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dest, 0, bytes)); // start from zero

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    cudaEvent_t start = get_time();
    add_kernel_interleaved<<<grid, block>>>(d_dest, d_src, iter, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t end = get_time();
    float ms = elapsed_time(start, end);

    CUDA_CHECK(cudaMemcpy(h_dest, d_dest, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dest));
    CUDA_CHECK(cudaFree(d_src));
    return ms;
}

float add_test_noninterleaved_gpu(NON_INTERLEAVED_T h_dest, NON_INTERLEAVED_T h_src,
                                  unsigned int iter, unsigned int num_elements) {
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

    CUDA_CHECK(cudaMemcpy(d_src.a, h_src.a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src.b, h_src.b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src.c, h_src.c, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src.d, h_src.d, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_dest.a, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dest.b, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dest.c, 0, bytes));
    CUDA_CHECK(cudaMemset(d_dest.d, 0, bytes));

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    cudaEvent_t start = get_time();
    add_kernel_noninterleaved<<<grid, block>>>(d_dest, d_src, iter, num_elements);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t end = get_time();
    float ms = elapsed_time(start, end);

    CUDA_CHECK(cudaMemcpy(h_dest.a, d_dest.a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.b, d_dest.b, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.c, d_dest.c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.d, d_dest.d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dest.a)); CUDA_CHECK(cudaFree(d_dest.b));
    CUDA_CHECK(cudaFree(d_dest.c)); CUDA_CHECK(cudaFree(d_dest.d));
    CUDA_CHECK(cudaFree(d_src.a));  CUDA_CHECK(cudaFree(d_src.b));
    CUDA_CHECK(cudaFree(d_src.c));  CUDA_CHECK(cudaFree(d_src.d));

    return ms;
}

// -----------------------------------------------------------------------------
// Bitreverse: same as original, with added kernel experiments
// -----------------------------------------------------------------------------
__host__ __device__ unsigned int bitreverse(unsigned int number) {
    number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
    number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
    number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
    return number;
}

__global__ void bitreverse_kernel(unsigned int *data, unsigned int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        data[tid] = bitreverse(data[tid]);
    }
}

// -----------------------------------------------------------------------------
// Main function: Run experiments and output CSV data
// -----------------------------------------------------------------------------
int main(void) {
    printf("=== Global Memory Experiments: Interleaved vs Non-Interleaved ===\n");
    printf("CSV Format: num_elements,iterations,cpu_interleaved_ms,gpu_interleaved_ms,cpu_noninterleaved_ms,gpu_noninterleaved_ms\n");

    // Test over a range of sizes and iteration counts
    unsigned int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    unsigned int iters[] = {1, 2, 4, 8, 16, 32};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iters = sizeof(iters) / sizeof(iters[0]);

    for (int s = 0; s < num_sizes; s++) {
        unsigned int num_elements = sizes[s];
        // Allocate and initialize interleaved arrays
        INTERLEAVED_T *src_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        INTERLEAVED_T *dest_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        // Allocate and initialize non-interleaved arrays
        NON_INTERLEAVED_T src_n, dest_n;
        src_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        src_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        src_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        src_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        dest_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        dest_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        dest_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        dest_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));

        // Initialize data for both tests
        for (unsigned int i = 0; i < num_elements; i++) {
            src_i[i].a = i; src_i[i].b = i+1; src_i[i].c = i+2; src_i[i].d = i+3;
            dest_i[i].a = 0; dest_i[i].b = 0; dest_i[i].c = 0; dest_i[i].d = 0;
            src_n.a[i] = i; src_n.b[i] = i+1; src_n.c[i] = i+2; src_n.d[i] = i+3;
            dest_n.a[i] = 0; dest_n.b[i] = 0; dest_n.c[i] = 0; dest_n.d[i] = 0;
        }

        for (int it = 0; it < num_iters; it++) {
            unsigned int iter = iters[it];

            float cpu_int = add_test_interleaved_cpu(dest_i, src_i, iter, num_elements);
            // Reset dest_i for fair GPU testing
            for (unsigned int i = 0; i < num_elements; i++) {
                dest_i[i].a = 0; dest_i[i].b = 0;
                dest_i[i].c = 0; dest_i[i].d = 0;
            }
            float gpu_int = add_test_interleaved_gpu(dest_i, src_i, iter, num_elements);

            float cpu_nint = add_test_noninterleaved_cpu(dest_n, src_n, iter, num_elements);
            // Reset dest_n arrays
            for (unsigned int i = 0; i < num_elements; i++) {
                dest_n.a[i] = 0; dest_n.b[i] = 0;
                dest_n.c[i] = 0; dest_n.d[i] = 0;
            }
            float gpu_nint = add_test_noninterleaved_gpu(dest_n, src_n, iter, num_elements);

            printf("%u,%u,%.4f,%.4f,%.4f,%.4f\n",
                   num_elements, iter, cpu_int, gpu_int, cpu_nint, gpu_nint);
        }
        free(src_i); free(dest_i);
        free(src_n.a); free(src_n.b); free(src_n.c); free(src_n.d);
        free(dest_n.a); free(dest_n.b); free(dest_n.c); free(dest_n.d);
    }

    // -------------------------------------------------------------------------
    // Extra experiment: Bitreverse with varying block sizes (based on original)
    // -------------------------------------------------------------------------
    printf("\n=== Bitreverse Experiments ===\n");
    printf("CSV Format: arraySize,blockSize,bitreverseTime_ms\n");
    unsigned int arraySize = 1 << 16; // 65536 elements
    unsigned int *h_data = (unsigned int*) malloc(arraySize * sizeof(unsigned int));
    for (unsigned int i = 0; i < arraySize; i++) {
        h_data[i] = i;
    }
    unsigned int *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, arraySize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice));
    for (unsigned int blockSize = 64; blockSize <= 1024; blockSize *= 2) {
        dim3 block(blockSize);
        dim3 grid((arraySize + blockSize - 1) / blockSize);
        CUDA_CHECK(cudaMemcpy(d_data, h_data, arraySize * sizeof(unsigned int), cudaMemcpyHostToDevice));
        cudaEvent_t start = get_time();
        bitreverse_kernel<<<grid, block>>>(d_data, arraySize);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t end = get_time();
        float time_ms = elapsed_time(start, end);
        printf("%u,%u,%.4f\n", arraySize, blockSize, time_ms);
    }
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
