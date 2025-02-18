/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Modified for improved timing/logging and reduced console output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// New macro for error checking
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(err);                                             \
    }                                                          \
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
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

// Removed per-thread debug prints from this CPU test.
__host__ float add_test_interleaved_cpu(
        INTERLEAVED_T *host_dest_ptr,
        const INTERLEAVED_T *host_src_ptr,
        const unsigned int iter,
        const unsigned int num_elements) {
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
    cudaEventSynchronize(end_time);
    float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);
    return delta;
}

__global__ void add_kernel_interleaved(INTERLEAVED_T *dest_ptr,
        const INTERLEAVED_T *src_ptr, const unsigned int iter,
        const unsigned int num_elements) {

    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements)
    {
        for(unsigned int i = 0; i < iter; i++) {
            dest_ptr[tid].a += src_ptr[tid].a;
            dest_ptr[tid].b += src_ptr[tid].b;
            dest_ptr[tid].c += src_ptr[tid].c;
            dest_ptr[tid].d += src_ptr[tid].d;
        }
    }
}

__host__ float add_test_interleaved(INTERLEAVED_T *host_dest_ptr,
        const INTERLEAVED_T *host_src_ptr, const unsigned int iter,
        const unsigned int num_elements)
{
    const unsigned int num_threads = 256;
    const unsigned int num_blocks = (num_elements + (num_threads-1)) / num_threads;
    const size_t num_bytes = sizeof(INTERLEAVED_T) * num_elements;

    INTERLEAVED_T *device_dest_ptr;
    INTERLEAVED_T *device_src_ptr;

    CUDA_CHECK(cudaMalloc((void **) &device_src_ptr, num_bytes));
    CUDA_CHECK(cudaMalloc((void **) &device_dest_ptr, num_bytes));

    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaStream_t test_stream;
    cudaStreamCreate(&test_stream);

    CUDA_CHECK(cudaMemcpy(device_src_ptr, host_src_ptr, num_bytes, cudaMemcpyHostToDevice));

    cudaEventRecord(kernel_start, test_stream);
    add_kernel_interleaved<<<num_blocks, num_threads, 0, test_stream>>>(device_dest_ptr, device_src_ptr, iter, num_elements);
    cudaEventRecord(kernel_stop, test_stream);
    cudaEventSynchronize(kernel_stop);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, kernel_start, kernel_stop);

    CUDA_CHECK(cudaFree(device_src_ptr));
    CUDA_CHECK(cudaFree(device_dest_ptr));
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaStreamDestroy(test_stream);

    return delta;
}

// ----------------------------
// Bitreverse kernel and helper
// ----------------------------
__host__ __device__ unsigned int bitreverse(unsigned int number) {
    number = ((0xf0f0f0f0u & number) >> 4) | ((0x0f0f0f0fu & number) << 4);
    number = ((0xccccccccu & number) >> 2) | ((0x33333333u & number) << 2);
    number = ((0xaaaaaaaau & number) >> 1) | ((0x55555555u & number) << 1);
    return number;
}

__global__ void bitreverse_kernel(void *data) {
    unsigned int *idata = (unsigned int*) data;
    idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

// ----------------------------
// Experiment: Bitreverse over various sizes
// ----------------------------
void experiment_bitreverse()
{
    // Different experiment sizes
    const unsigned int sizes[] = {64, 128, 256, 512, 1024};
    const int num_experiments = sizeof(sizes)/sizeof(sizes[0]);

    printf("Experiment,Size,KernelTime(ms),SampleOutput\n");
    for (int exp = 0; exp < num_experiments; exp++) {
        unsigned int size = sizes[exp];
        unsigned int *idata = new unsigned int[size];
        unsigned int *odata = new unsigned int[size];

        for (unsigned int i = 0; i < size; i++) {
            idata[i] = i;
        }
        void *d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, sizeof(unsigned int) * size));
        CUDA_CHECK(cudaMemcpy(d, idata, sizeof(unsigned int) * size, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));

        bitreverse_kernel<<<1, size>>>(d);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float t = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&t, start, stop));

        // Copy a few output elements to verify correctness.
        CUDA_CHECK(cudaMemcpy(odata, d, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost));
        // Build a sample string from the first 3 outputs.
        char sampleOutput[128];
        snprintf(sampleOutput, sizeof(sampleOutput), "%u %u %u",
                 odata[0], (size>1?odata[1]:0), (size>2?odata[2]:0));

        // Log as CSV: experiment number, size, kernel time, sample output.
        printf("%d,%u,%.3f,%s\n", exp, size, t, sampleOutput);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d));
        delete [] idata;
        delete [] odata;
    }
}

// ----------------------------
// Host-side test function wrappers
// ----------------------------
void execute_host_functions()
{
    INTERLEAVED_T host_dest[NUM_ELEMENTS] = {0};
    INTERLEAVED_T host_src[NUM_ELEMENTS] = {0};

    // Experiment with different iteration counts
    const unsigned int iterations[] = {1, 4, 8};
    const int num_tests = sizeof(iterations)/sizeof(iterations[0]);

    printf("Interleaved CPU Addition Timing (CSV): TestIter,NumElements,Time(ms)\n");
    for (int t = 0; t < num_tests; t++) {
        float duration = add_test_interleaved_cpu(host_dest, host_src, iterations[t], NUM_ELEMENTS);
        printf("%u,%d,%.3f\n", iterations[t], NUM_ELEMENTS, duration);
    }
}

void execute_gpu_functions()
{
    // Run the bitreverse kernel experiment over several sizes.
    experiment_bitreverse();
}

int main(void) {
    // Run CPU tests and GPU experiments (timing/logging only)
    execute_host_functions();
    execute_gpu_functions();

    return 0;
}
