/******************************************************************************
 * global_memory.cu
 *
 * Revised experiments for interleaved vs. non-interleaved memory accesses.
 * This version runs tests for array sizes from 256 up to 131072 (without 
 * exceeding 262144) and for several iteration counts.
 *
 * Build with: nvcc -O2 -o global_memory.exe global_memory.cu
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

// Define minimum and maximum number of elements for experiments
#define MIN_ELEMENTS 256
#define MAX_ELEMENTS 131072  // Do not exceed 262144 to avoid segfaults

// Iteration counts to test
const unsigned int iterationsArray[] = {1, 2, 4, 8, 16, 32};

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

// Timing helpers using CUDA events
__host__ cudaEvent_t get_time(void) {
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time, 0);
    cudaEventSynchronize(time);
    return time;
}

__host__ float elapsed_time(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
}

// CPU interleaved addition: loops over elements and iterations
__host__ float cpu_add_interleaved(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                                   unsigned int num_elements, unsigned int iter) {
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

// CPU non-interleaved addition: operates on 4 separate arrays
__host__ float cpu_add_noninterleaved(NON_INTERLEAVED_T *dest, NON_INTERLEAVED_T *src,
                                      unsigned int num_elements, unsigned int iter) {
    cudaEvent_t start = get_time();
    for (unsigned int i = 0; i < num_elements; i++) {
        for (unsigned int j = 0; j < iter; j++) {
            dest->a[i] += src->a[i];
            dest->b[i] += src->b[i];
            dest->c[i] += src->c[i];
            dest->d[i] += src->d[i];
        }
    }
    cudaEvent_t end = get_time();
    return elapsed_time(start, end);
}

// GPU kernel for interleaved addition
__global__ void gpu_add_interleaved_kernel(INTERLEAVED_T *dest, const INTERLEAVED_T *src,
                                             unsigned int num_elements, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_elements) {
        for (unsigned int i = 0; i < iter; i++) {
            dest[tid].a += src[tid].a;
            dest[tid].b += src[tid].b;
            dest[tid].c += src[tid].c;
            dest[tid].d += src[tid].d;
        }
    }
}

// GPU kernel for non-interleaved addition
__global__ void gpu_add_noninterleaved_kernel(NON_INTERLEAVED_T dest, NON_INTERLEAVED_T src,
                                                unsigned int num_elements, unsigned int iter) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_elements) {
        for (unsigned int i = 0; i < iter; i++) {
            dest.a[tid] += src.a[tid];
            dest.b[tid] += src.b[tid];
            dest.c[tid] += src.c[tid];
            dest.d[tid] += src.d[tid];
        }
    }
}

// GPU interleaved addition function
__host__ float gpu_add_interleaved(INTERLEAVED_T *host_dest, const INTERLEAVED_T *host_src,
                                   unsigned int num_elements, unsigned int iter) {
    size_t bytes = num_elements * sizeof(INTERLEAVED_T);
    INTERLEAVED_T *d_dest = NULL, *d_src = NULL;
    cudaMalloc((void**)&d_dest, bytes);
    cudaMalloc((void**)&d_src, bytes);

    cudaMemcpy(d_src, host_src, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_dest, 0, bytes);

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    cudaEvent_t start = get_time();
    gpu_add_interleaved_kernel<<<grid, block>>>(d_dest, d_src, num_elements, iter);
    cudaDeviceSynchronize();
    cudaEvent_t end = get_time();

    float ms = elapsed_time(start, end);
    cudaMemcpy(host_dest, d_dest, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_dest);
    cudaFree(d_src);
    return ms;
}

// GPU non-interleaved addition function
__host__ float gpu_add_noninterleaved(NON_INTERLEAVED_T *host_dest, NON_INTERLEAVED_T *host_src,
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

    cudaMemcpy(d_src.a, host_src->a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src.b, host_src->b, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src.c, host_src->c, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_src.d, host_src->d, bytes, cudaMemcpyHostToDevice);

    cudaMemset(d_dest.a, 0, bytes);
    cudaMemset(d_dest.b, 0, bytes);
    cudaMemset(d_dest.c, 0, bytes);
    cudaMemset(d_dest.d, 0, bytes);

    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    cudaEvent_t start = get_time();
    gpu_add_noninterleaved_kernel<<<grid, block>>>(d_dest, d_src, num_elements, iter);
    cudaDeviceSynchronize();
    cudaEvent_t end = get_time();

    float ms = elapsed_time(start, end);
    cudaMemcpy(host_dest->a, d_dest.a, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dest->b, d_dest.b, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dest->c, d_dest.c, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_dest->d, d_dest.d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_dest.a); cudaFree(d_dest.b); cudaFree(d_dest.c); cudaFree(d_dest.d);
    cudaFree(d_src.a);  cudaFree(d_src.b);  cudaFree(d_src.c);  cudaFree(d_src.d);

    return ms;
}

// Global bitreverse kernel (same as the original)
__global__ void bitreverse_kernel(unsigned int *data, unsigned int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        unsigned int x = data[tid];
        x = ((0xf0f0f0f0 & x) >> 4) | ((0x0f0f0f0f & x) << 4);
        x = ((0xcccccccc & x) >> 2) | ((0x33333333 & x) << 2);
        x = ((0xaaaaaaaa & x) >> 1) | ((0x55555555 & x) << 1);
        data[tid] = x;
    }
}

int main(void) {
    printf("=== Global Memory Experiments: Interleaved vs Non-Interleaved ===\n");
    printf("CSV Format: num_elements,iterations,cpu_interleaved_ms,gpu_interleaved_ms,cpu_noninterleaved_ms,gpu_noninterleaved_ms\n");
    
    // Test array sizes from MIN_ELEMENTS to MAX_ELEMENTS doubling each time.
    for (unsigned int num_elements = MIN_ELEMENTS; num_elements <= MAX_ELEMENTS; num_elements *= 2) {
        // Allocate host arrays for interleaved layout
        INTERLEAVED_T *host_src_inter = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        INTERLEAVED_T *host_dest_inter = (INTERLEAVED_T*) malloc(num_elements * sizeof(INTERLEAVED_T));
        
        // Allocate host arrays for non-interleaved layout
        NON_INTERLEAVED_T host_src_non, host_dest_non;
        host_src_non.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_non.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_non.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_src_non.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_non.a = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_non.b = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_non.c = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        host_dest_non.d = (unsigned int*) malloc(num_elements * sizeof(unsigned int));
        
        // Initialize both sets of arrays with sample data
        for (unsigned int i = 0; i < num_elements; i++) {
            host_src_inter[i].a = i;
            host_src_inter[i].b = i + 1;
            host_src_inter[i].c = i + 2;
            host_src_inter[i].d = i + 3;
            host_dest_inter[i].a = 0;
            host_dest_inter[i].b = 0;
            host_dest_inter[i].c = 0;
            host_dest_inter[i].d = 0;
            
            host_src_non.a[i] = i;
            host_src_non.b[i] = i + 1;
            host_src_non.c[i] = i + 2;
            host_src_non.d[i] = i + 3;
            host_dest_non.a[i] = 0;
            host_dest_non.b[i] = 0;
            host_dest_non.c[i] = 0;
            host_dest_non.d[i] = 0;
        }
        
        // Loop over the iteration counts
        for (unsigned int it = 0; it < sizeof(iterationsArray)/sizeof(iterationsArray[0]); it++) {
            unsigned int iter = iterationsArray[it];
            
            float cpu_int_time = cpu_add_interleaved(host_dest_inter, host_src_inter, num_elements, iter);
            // Reset interleaved destination array to zero
            for (unsigned int i = 0; i < num_elements; i++) {
                host_dest_inter[i].a = 0;
                host_dest_inter[i].b = 0;
                host_dest_inter[i].c = 0;
                host_dest_inter[i].d = 0;
            }
            float gpu_int_time = gpu_add_interleaved(host_dest_inter, host_src_inter, num_elements, iter);
            
            float cpu_nint_time = cpu_add_noninterleaved(&host_dest_non, &host_src_non, num_elements, iter);
            // Reset non-interleaved destination arrays to zero
            for (unsigned int i = 0; i < num_elements; i++) {
                host_dest_non.a[i] = 0;
                host_dest_non.b[i] = 0;
                host_dest_non.c[i] = 0;
                host_dest_non.d[i] = 0;
            }
            float gpu_nint_time = gpu_add_noninterleaved(&host_dest_non, &host_src_non, num_elements, iter);
            
            printf("%u,%u,%.4f,%.4f,%.4f,%.4f\n",
                   num_elements, iter, cpu_int_time, gpu_int_time, cpu_nint_time, gpu_nint_time);
        }
        
        free(host_src_inter);
        free(host_dest_inter);
        free(host_src_non.a); free(host_src_non.b);
        free(host_src_non.c); free(host_src_non.d);
        free(host_dest_non.a); free(host_dest_non.b);
        free(host_dest_non.c); free(host_dest_non.d);
    }
    
    // Bitreverse experiment section
    printf("\n=== Bitreverse Experiments ===\n");
    printf("CSV Format: arraySize,blockSize,bitreverseTime_ms\n");
    
    const unsigned int BITREVERSE_SIZE = 1 << 16; // 65536 elements
    unsigned int *h_data = (unsigned int*) malloc(BITREVERSE_SIZE * sizeof(unsigned int));
    for (unsigned int i = 0; i < BITREVERSE_SIZE; i++) {
        h_data[i] = i;
    }
    unsigned int *d_data = NULL;
    cudaMalloc(&d_data, BITREVERSE_SIZE * sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, BITREVERSE_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    for (unsigned int blockSize = 64; blockSize <= 1024; blockSize *= 2) {
        dim3 block(blockSize);
        dim3 grid((BITREVERSE_SIZE + blockSize - 1) / blockSize);
        // Reset device data for each experiment
        cudaMemcpy(d_data, h_data, BITREVERSE_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaEvent_t start = get_time();
        bitreverse_kernel<<<grid, block>>>(d_data, BITREVERSE_SIZE);
        cudaDeviceSynchronize();
        cudaEvent_t end = get_time();
        float time_ms = elapsed_time(start, end);
        printf("%u,%u,%.4f\n", BITREVERSE_SIZE, blockSize, time_ms);
    }
    
    cudaFree(d_data);
    free(h_data);
    cudaDeviceReset();
    return 0;
}
