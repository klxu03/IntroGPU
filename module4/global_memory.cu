/******************************************************************************
 * global_memory_experiments.cu
 *
 * Demonstrates interleaved vs. non-interleaved memory access on CPU and GPU
 * with larger array sizes (up to 262144 elements) and various iteration counts.
 * CPU timings are measured using clock_gettime for better resolution of linear scaling.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <assert.h>

#define MAX_SIZE 262144  // Maximum number of elements to avoid segfaults

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

// CPU timing helper using clock_gettime
double cpu_time_diff(struct timespec start, struct timespec end) {
    double start_ms = start.tv_sec * 1000.0 + start.tv_nsec / 1e6;
    double end_ms   = end.tv_sec   * 1000.0 + end.tv_nsec   / 1e6;
    return end_ms - start_ms;
}

// CPU addition for interleaved data (measured with clock_gettime)
double cpu_add_interleaved(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                           unsigned int num_elements, unsigned int iter) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (unsigned int i = 0; i < num_elements; i++) {
        for (unsigned int j = 0; j < iter; j++) {
            dest[i].a += src[i].a;
            dest[i].b += src[i].b;
            dest[i].c += src[i].c;
            dest[i].d += src[i].d;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    return cpu_time_diff(start, end);
}

// CPU addition for non-interleaved data (measured with clock_gettime)
double cpu_add_noninterleaved(NON_INTERLEAVED_T *dest, const NON_INTERLEAVED_T *src,
                              unsigned int num_elements, unsigned int iter) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (unsigned int i = 0; i < num_elements; i++) {
        for (unsigned int j = 0; j < iter; j++) {
            dest->a[i] += src->a[i];
            dest->b[i] += src->b[i];
            dest->c[i] += src->c[i];
            dest->d[i] += src->d[i];
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    return cpu_time_diff(start, end);
}

// GPU kernel for interleaved addition
__global__ void add_kernel_interleaved(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                                       unsigned int num_elements, unsigned int iter) {
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

// GPU kernel for non-interleaved addition
__global__ void add_kernel_noninterleaved(NON_INTERLEAVED_T dest, NON_INTERLEAVED_T src,
                                          unsigned int num_elements, unsigned int iter) {
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

// GPU addition for interleaved data (using CUDA events)
double gpu_add_interleaved(INTERLEAVED_T *h_dest, const INTERLEAVED_T *h_src,
                           unsigned int num_elements, unsigned int iter) {
    size_t bytes = num_elements * sizeof(INTERLEAVED_T);
    INTERLEAVED_T *d_dest, *d_src;
    cudaMalloc((void**)&d_dest, bytes);
    cudaMalloc((void**)&d_src, bytes);
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_dest, 0, bytes);

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_kernel_interleaved<<<grid, block>>>(d_dest, d_src, num_elements, iter);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_dest, d_dest, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dest);
    cudaFree(d_src);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms;
}

// GPU addition for non-interleaved data (using CUDA events)
double gpu_add_noninterleaved(NON_INTERLEAVED_T *h_dest, const NON_INTERLEAVED_T *h_src,
                              unsigned int num_elements, unsigned int iter) {
    size_t bytes = num_elements * sizeof(unsigned int);
    NON_INTERLEAVED_T d_dest, d_src;
    cudaMalloc((void**)&d_dest.a, bytes);
    cudaMalloc((void**)&d_dest.b, bytes);
    cudaMalloc((void**)&d_dest.c, bytes);
    cudaMalloc((void**)&d_dest.d, bytes);
    cudaMalloc((void**)&d_src.a, bytes);
    cudaMalloc((void**)&d_src.b, bytes);
    cudaMalloc((void**)&d_src.c, bytes);
    cudaMalloc((void**)&d_src.d, bytes);
    cudaMemcpy(d_src.a, h_src->a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src.b, h_src->b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src.c, h_src->c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src.d, h_src->d, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_dest.a, 0, bytes);
    cudaMemset(d_dest.b, 0, bytes);
    cudaMemset(d_dest.c, 0, bytes);
    cudaMemset(d_dest.d, 0, bytes);

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    add_kernel_noninterleaved<<<grid, block>>>(d_dest, d_src, num_elements, iter);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_dest->a, d_dest.a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dest->b, d_dest.b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dest->c, d_dest.c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dest->d, d_dest.d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_dest.a); cudaFree(d_dest.b); cudaFree(d_dest.c); cudaFree(d_dest.d);
    cudaFree(d_src.a); cudaFree(d_src.b); cudaFree(d_src.c); cudaFree(d_src.d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return (double)ms;
}

int main(void) {
    printf("=== Global Memory Experiments: Interleaved vs Non-Interleaved ===\n");
    printf("CSV Format: num_elements,iterations,cpu_interleaved_ms,gpu_interleaved_ms,cpu_noninterleaved_ms,gpu_noninterleaved_ms\n");

    unsigned int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, MAX_SIZE};
    unsigned int iters[]  = {1, 2, 4, 8, 16, 32};

    for (unsigned int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        unsigned int num_elements = sizes[s];

        // Allocate and initialize interleaved data on host
        INTERLEAVED_T *h_src_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        INTERLEAVED_T *h_dest_i = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        // Allocate and initialize non-interleaved data on host
        NON_INTERLEAVED_T h_src_n, h_dest_n;
        h_src_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_src_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_src_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_src_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_dest_n.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_dest_n.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_dest_n.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        h_dest_n.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));

        for (unsigned int i = 0; i < num_elements; i++) {
            h_src_i[i].a = i;
            h_src_i[i].b = i + 1;
            h_src_i[i].c = i + 2;
            h_src_i[i].d = i + 3;
            h_dest_i[i].a = 0;
            h_dest_i[i].b = 0;
            h_dest_i[i].c = 0;
            h_dest_i[i].d = 0;

            h_src_n.a[i] = i;
            h_src_n.b[i] = i + 1;
            h_src_n.c[i] = i + 2;
            h_src_n.d[i] = i + 3;
            h_dest_n.a[i] = 0;
            h_dest_n.b[i] = 0;
            h_dest_n.c[i] = 0;
            h_dest_n.d[i] = 0;
        }

        for (unsigned int it = 0; it < sizeof(iters)/sizeof(iters[0]); it++) {
            unsigned int iter = iters[it];

            double cpu_int_time = cpu_add_interleaved(h_dest_i, h_src_i, num_elements, iter);
            // Reset destination for a fair test
            for (unsigned int i = 0; i < num_elements; i++) {
                h_dest_i[i].a = 0;
                h_dest_i[i].b = 0;
                h_dest_i[i].c = 0;
                h_dest_i[i].d = 0;
            }
            double gpu_int_time = gpu_add_interleaved(h_dest_i, h_src_i, num_elements, iter);

            double cpu_nint_time = cpu_add_noninterleaved(&h_dest_n, &h_src_n, num_elements, iter);
            for (unsigned int i = 0; i < num_elements; i++) {
                h_dest_n.a[i] = 0;
                h_dest_n.b[i] = 0;
                h_dest_n.c[i] = 0;
                h_dest_n.d[i] = 0;
            }
            double gpu_nint_time = gpu_add_noninterleaved(&h_dest_n, &h_src_n, num_elements, iter);

            printf("%u,%u,%.4f,%.4f,%.4f,%.4f\n",
                   num_elements, iter, cpu_int_time, gpu_int_time, cpu_nint_time, gpu_nint_time);
        }

        free(h_src_i); free(h_dest_i);
        free(h_src_n.a); free(h_src_n.b); free(h_src_n.c); free(h_src_n.d);
        free(h_dest_n.a); free(h_dest_n.b); free(h_dest_n.c); free(h_dest_n.d);
    }
    cudaDeviceReset();
    return 0;
}
