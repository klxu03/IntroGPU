/******************************************************************************
 * global_memory.cu
 *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Revised to fix a bug in the non-interleaved GPU kernel.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));         \
        exit(err);                                                    \
    }                                                                 \
} while(0)

static const int WORK_SIZE = 256;
#define NUM_ELEMENTS 4096

typedef struct {
    unsigned int a;
    unsigned int b;
    unsigned int c;
    unsigned int d;
} INTERLEAVED_T;

typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[NUM_ELEMENTS];

typedef unsigned int ARRAY_MEMBER_T[NUM_ELEMENTS];

typedef struct {
    ARRAY_MEMBER_T a;
    ARRAY_MEMBER_T b;
    ARRAY_MEMBER_T c;
    ARRAY_MEMBER_T d;
} NON_INTERLEAVED_T;

__host__ cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    CUDA_CHECK(cudaEventCreate(&time));
    CUDA_CHECK(cudaEventRecord(time, 0));
    return time;
}

__host__ float add_test_non_interleaved_cpu(
        NON_INTERLEAVED_T host_dest_ptr,
        NON_INTERLEAVED_T const host_src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    cudaEvent_t start_time = get_time();

    for (unsigned int tid = 0; tid < num_elements; tid++) {
        for (unsigned int i = 0; i < iter; i++) {
            host_dest_ptr.a[tid] += host_src_ptr.a[tid];
            host_dest_ptr.b[tid] += host_src_ptr.b[tid];
            host_dest_ptr.c[tid] += host_src_ptr.c[tid];
            host_dest_ptr.d[tid] += host_src_ptr.d[tid];
        }
    }

    cudaEvent_t end_time = get_time();
    CUDA_CHECK(cudaEventSynchronize(end_time));

    float delta = 0;
    CUDA_CHECK(cudaEventElapsedTime(&delta, start_time, end_time));

    return delta;
}

__host__ float add_test_interleaved_cpu(INTERLEAVED_T * const host_dest_ptr,
        const INTERLEAVED_T * const host_src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    cudaEvent_t start_time = get_time();
    for (unsigned int tid = 0; tid < num_elements; tid++) {
        for (unsigned int i = 0; i < iter; i++) {
            host_dest_ptr[tid].a += host_src_ptr[tid].a;
            host_dest_ptr[tid].b += host_src_ptr[tid].b;
            host_dest_ptr[tid].c += host_src_ptr[tid].c;
            host_dest_ptr[tid].d += host_src_ptr[tid].d;
        }
    }

    cudaEvent_t end_time = get_time();
    CUDA_CHECK(cudaEventSynchronize(end_time));

    float delta = 0;
    CUDA_CHECK(cudaEventElapsedTime(&delta, start_time, end_time));

    return delta;
}

__global__ void add_kernel_interleaved(INTERLEAVED_T * const dest_ptr,
        const INTERLEAVED_T * const src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_elements)
    {
        for (unsigned int i = 0; i < iter; i++) {
            dest_ptr[tid].a += src_ptr[tid].a;
            dest_ptr[tid].b += src_ptr[tid].b;
            dest_ptr[tid].c += src_ptr[tid].c;
            dest_ptr[tid].d += src_ptr[tid].d;
        }
    }
}

// --- Corrected Non-Interleaved GPU Kernel ---
// Each thread computes its own index and processes only that element.
__global__ void add_kernel_non_interleaved(
        NON_INTERLEAVED_T * const dest_ptr,
        NON_INTERLEAVED_T * const src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements)
    {
        for (unsigned int i = 0; i < iter; i++) {
            dest_ptr->a[tid] += src_ptr->a[tid];
            dest_ptr->b[tid] += src_ptr->b[tid];
            dest_ptr->c[tid] += src_ptr->c[tid];
            dest_ptr->d[tid] += src_ptr->d[tid];
        }
    }
}

__host__ float add_test_interleaved(INTERLEAVED_T * const host_dest_ptr,
        const INTERLEAVED_T * const host_src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    const unsigned int num_threads = 256;
    const unsigned int num_blocks = (num_elements + (num_threads - 1)) / num_threads;
    const size_t num_bytes = sizeof(INTERLEAVED_T) * num_elements;
    INTERLEAVED_T *device_dest_ptr;
    INTERLEAVED_T *device_src_ptr;

    CUDA_CHECK(cudaMalloc((void **) &device_src_ptr, num_bytes));
    CUDA_CHECK(cudaMalloc((void **) &device_dest_ptr, num_bytes));

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    cudaStream_t test_stream;
    CUDA_CHECK(cudaStreamCreate(&test_stream));

    CUDA_CHECK(cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(kernel_start, 0));

    add_kernel_interleaved<<<num_blocks, num_threads>>>(device_dest_ptr, device_src_ptr, iter, num_elements);

    CUDA_CHECK(cudaEventRecord(kernel_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float delta = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));

    CUDA_CHECK(cudaFree(device_src_ptr));
    CUDA_CHECK(cudaFree(device_dest_ptr));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaStreamDestroy(test_stream));

    return delta;
}

// --- New function for non-interleaved GPU test ---
// This function converts the interleaved host data into non-interleaved form,
// launches the fixed non-interleaved kernel, and returns the elapsed time.
__host__ float add_test_noninterleaved(INTERLEAVED_T * const host_dest_ptr,
        const INTERLEAVED_T * const host_src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    size_t num_bytes = num_elements * sizeof(unsigned int);

    // Convert interleaved data to non-interleaved format
    NON_INTERLEAVED_T h_src, h_dest;
    h_src.a = (unsigned int*) malloc(num_bytes);
    h_src.b = (unsigned int*) malloc(num_bytes);
    h_src.c = (unsigned int*) malloc(num_bytes);
    h_src.d = (unsigned int*) malloc(num_bytes);
    h_dest.a = (unsigned int*) malloc(num_bytes);
    h_dest.b = (unsigned int*) malloc(num_bytes);
    h_dest.c = (unsigned int*) malloc(num_bytes);
    h_dest.d = (unsigned int*) malloc(num_bytes);

    for (unsigned int i = 0; i < num_elements; i++) {
        h_src.a[i] = host_src_ptr[i].a;
        h_src.b[i] = host_src_ptr[i].b;
        h_src.c[i] = host_src_ptr[i].c;
        h_src.d[i] = host_src_ptr[i].d;
        h_dest.a[i] = 0;
        h_dest.b[i] = 0;
        h_dest.c[i] = 0;
        h_dest.d[i] = 0;
    }

    const unsigned int num_threads = 256;
    const unsigned int num_blocks = (num_elements + num_threads - 1) / num_threads;

    // Allocate device memory for the NON_INTERLEAVED_T structure and its arrays
    NON_INTERLEAVED_T *device_src;
    NON_INTERLEAVED_T *device_dest;
    CUDA_CHECK(cudaMalloc((void **)&device_src, sizeof(NON_INTERLEAVED_T)));
    CUDA_CHECK(cudaMalloc((void **)&device_dest, sizeof(NON_INTERLEAVED_T)));

    unsigned int *d_a_src, *d_b_src, *d_c_src, *d_d_src;
    unsigned int *d_a_dest, *d_b_dest, *d_c_dest, *d_d_dest;
    CUDA_CHECK(cudaMalloc((void**)&d_a_src, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b_src, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c_src, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_d_src, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_a_dest, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b_dest, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c_dest, num_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_d_dest, num_bytes));

    // Copy host data to device arrays
    CUDA_CHECK(cudaMemcpy(d_a_src, h_src.a, num_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_src, h_src.b, num_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_src, h_src.c, num_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_src, h_src.d, num_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_a_dest, 0, num_bytes));
    CUDA_CHECK(cudaMemset(d_b_dest, 0, num_bytes));
    CUDA_CHECK(cudaMemset(d_c_dest, 0, num_bytes));
    CUDA_CHECK(cudaMemset(d_d_dest, 0, num_bytes));

    // Set up the NON_INTERLEAVED_T structure on the host to point to device arrays
    NON_INTERLEAVED_T h_device_src, h_device_dest;
    h_device_src.a = d_a_src;
    h_device_src.b = d_b_src;
    h_device_src.c = d_c_src;
    h_device_src.d = d_d_src;
    h_device_dest.a = d_a_dest;
    h_device_dest.b = d_b_dest;
    h_device_dest.c = d_c_dest;
    h_device_dest.d = d_d_dest;

    CUDA_CHECK(cudaMemcpy(device_src, &h_device_src, sizeof(NON_INTERLEAVED_T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_dest, &h_device_dest, sizeof(NON_INTERLEAVED_T), cudaMemcpyHostToDevice));

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start, 0));

    add_kernel_non_interleaved<<<num_blocks, num_threads>>>(device_dest, device_src, iter, num_elements);

    CUDA_CHECK(cudaEventRecord(kernel_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float delta = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));

    // Optionally copy results back from device if verification is needed
    CUDA_CHECK(cudaMemcpy(h_dest.a, d_a_dest, num_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.b, d_b_dest, num_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.c, d_c_dest, num_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_dest.d, d_d_dest, num_bytes, cudaMemcpyDeviceToHost));

    // Cleanup
    free(h_src.a); free(h_src.b); free(h_src.c); free(h_src.d);
    free(h_dest.a); free(h_dest.b); free(h_dest.c); free(h_dest.d);
    CUDA_CHECK(cudaFree(d_a_src)); CUDA_CHECK(cudaFree(d_b_src));
    CUDA_CHECK(cudaFree(d_c_src)); CUDA_CHECK(cudaFree(d_d_src));
    CUDA_CHECK(cudaFree(d_a_dest)); CUDA_CHECK(cudaFree(d_b_dest));
    CUDA_CHECK(cudaFree(d_c_dest)); CUDA_CHECK(cudaFree(d_d_dest));
    CUDA_CHECK(cudaFree(device_src)); CUDA_CHECK(cudaFree(device_dest));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    return delta;
}

__host__ float select_samples_cpu(unsigned int * const sample_data,
        const unsigned int sample_interval,
        const unsigned int num_elements,
        const unsigned int * const src_data)
{
    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start, 0));

    unsigned int sample_idx = 0;
    for (unsigned int src_idx = 0; src_idx < num_elements; src_idx += sample_interval) {
        sample_data[sample_idx] = src_data[src_idx];
        sample_idx++;
    }

    CUDA_CHECK(cudaEventRecord(kernel_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float delta = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));
    return delta;
}

__global__ void select_samples_gpu_kernel(unsigned int * const sample_data,
        const unsigned int sample_interval,
        const unsigned int * const src_data)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    sample_data[tid] = src_data[tid * sample_interval];
}

__host__ float select_samples_gpu(unsigned int * const sample_data,
        const unsigned int sample_interval,
        const unsigned int num_elements,
        const unsigned int num_samples,
        const unsigned int * const src_data,
        const unsigned int num_threads_per_block,
        const char * prefix)
{
    const unsigned int num_blocks = num_samples / num_threads_per_block;
    assert((num_blocks * num_threads_per_block) == num_samples);

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start, 0));

    select_samples_gpu_kernel<<<num_blocks, num_threads_per_block>>>(sample_data, sample_interval, src_data);

    CUDA_CHECK(cudaEventRecord(kernel_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float delta = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));
    return delta;
}

int compare_func (const void * a, const void * b)
{
    return (*(int*)a - *(int*)b);
}

__host__ float sort_samples_cpu(unsigned int * const sample_data,
        const unsigned int num_samples)
{
    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start, 0));

    qsort(sample_data, num_samples, sizeof(unsigned int), &compare_func);

    CUDA_CHECK(cudaEventRecord(kernel_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float delta = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&delta, kernel_start, kernel_stop));
    return delta;
}

__host__ __device__ unsigned int bitreverse(unsigned int number) {
    number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
    number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
    number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
    return number;
}

__global__ void bitreverse(void *data) {
    unsigned int *idata = (unsigned int*) data;
    idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

void execute_host_functions()
{
    INTERLEAVED_T host_dest_ptr[NUM_ELEMENTS];
    INTERLEAVED_T host_src_ptr[NUM_ELEMENTS];
    float duration = add_test_interleaved_cpu(host_dest_ptr, host_src_ptr, 4, NUM_ELEMENTS);
    printf("duration: %fms\n", duration);
}

void execute_gpu_functions()
{
    void *d = NULL;
    unsigned int idata[WORK_SIZE], odata[WORK_SIZE];
    for (int i = 0; i < WORK_SIZE; i++)
        idata[i] = (unsigned int) i;

    CUDA_CHECK(cudaMalloc((void**)&d, sizeof(int) * WORK_SIZE));
    CUDA_CHECK(cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

    bitreverse<<<1, WORK_SIZE, WORK_SIZE * sizeof(int)>>>(d);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

    for (int i = 0; i < WORK_SIZE; i++)
        printf("Input value: %u, device output: %u, host output: %u\n",
                idata[i], odata[i], bitreverse(idata[i]));

    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaDeviceReset());
}

int main(void) {
    printf("=== Global Memory Experiments: Interleaved vs Non-Interleaved ===\n");
    printf("CSV Format: num_elements,iterations,cpu_interleaved_ms,gpu_interleaved_ms,cpu_noninterleaved_ms,gpu_noninterleaved_ms\n");

    // Test array sizes from 256 to 262144
    unsigned int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    unsigned int iters[] = {1, 2, 4, 8, 16, 32};

    for (unsigned int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        unsigned int num_elements = sizes[s];

        INTERLEAVED_T *host_src_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        INTERLEAVED_T *host_dest_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));

        NON_INTERLEAVED_T host_src_n;
        NON_INTERLEAVED_T host_dest_n;
        host_src_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));

        for (unsigned int i = 0; i < num_elements; i++) {
            host_src_i[i].a = i;
            host_src_i[i].b = i + 1;
            host_src_i[i].c = i + 2;
            host_src_i[i].d = i + 3;
            host_src_n.a[i] = i;
            host_src_n.b[i] = i + 1;
            host_src_n.c[i] = i + 2;
            host_src_n.d[i] = i + 3;

            host_dest_i[i].a = 0;
            host_dest_i[i].b = 0;
            host_dest_i[i].c = 0;
            host_dest_i[i].d = 0;
            host_dest_n.a[i] = 0;
            host_dest_n.b[i] = 0;
            host_dest_n.c[i] = 0;
            host_dest_n.d[i] = 0;
        }

        for (unsigned int it = 0; it < sizeof(iters)/sizeof(iters[0]); it++) {
            unsigned int iter = iters[it];

            float cpu_int_ms = add_test_interleaved_cpu(host_dest_i, host_src_i, iter, num_elements);

            // Reset host_dest_i for GPU test
            for (unsigned int i = 0; i < num_elements; i++) {
                host_dest_i[i].a = 0;
                host_dest_i[i].b = 0;
                host_dest_i[i].c = 0;
                host_dest_i[i].d = 0;
            }

            float gpu_int_ms = add_test_interleaved(host_dest_i, host_src_i, iter, num_elements);
            float cpu_nint_ms = add_test_non_interleaved_cpu(host_dest_n, host_src_n, iter, num_elements);

            // Reset host_dest_n for GPU test
            for (unsigned int i = 0; i < num_elements; i++) {
                host_dest_n.a[i] = 0;
                host_dest_n.b[i] = 0;
                host_dest_n.c[i] = 0;
                host_dest_n.d[i] = 0;
            }

            float gpu_nint_ms = add_test_noninterleaved(host_dest_i, host_src_i, iter, num_elements);

            printf("%u,%u,%.4f,%.4f,%.4f,%.4f\n",
                   num_elements, iter, cpu_int_ms, gpu_int_ms, cpu_nint_ms, gpu_nint_ms);
        }

        free(host_src_i);
        free(host_dest_i);
        free(host_src_n.a); free(host_src_n.b); free(host_src_n.c); free(host_src_n.d);
        free(host_dest_n.a); free(host_dest_n.b); free(host_dest_n.c); free(host_dest_n.d);
    }

    printf("\n=== Bitreverse Experiments ===\n");
    printf("CSV Format: arraySize,blockSize,bitreverseTime_ms\n");

    const unsigned int BITREVERSE_SIZE = 1 << 16;
    unsigned int *h_data = (unsigned int*) malloc(BITREVERSE_SIZE * sizeof(unsigned int));
    for (unsigned int i = 0; i < BITREVERSE_SIZE; i++) {
        h_data[i] = i;
    }

    unsigned int *d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, BITREVERSE_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, BITREVERSE_SIZE * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    for (unsigned int blockSize = 64; blockSize <= 1024; blockSize *= 2) {
        dim3 block(blockSize);
        dim3 grid((BITREVERSE_SIZE + blockSize - 1) / blockSize);

        CUDA_CHECK(cudaMemcpy(d_data, h_data, BITREVERSE_SIZE * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        cudaEvent_t start = get_time();
        bitreverse<<<grid, block>>>(d_data);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t end = get_time();

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        printf("%u,%u,%.4f\n", BITREVERSE_SIZE, blockSize, ms);
    }

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    CUDA_CHECK(cudaDeviceReset());
    execute_host_functions();
    execute_gpu_functions();
    return 0;
}
