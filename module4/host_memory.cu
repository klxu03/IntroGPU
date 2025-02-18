/******************************************************************************
 * host_memory.cu
 *
 * Based on the original SAXPY sample from NVIDIA's devblogs and our earlier
 * experiments, this version performs SAXPY operations over a range of vector
 * sizes, timing both GPU and CPU executions and reporting results in CSV format.
 *
 * Additional timing measurements have been added to help analyze performance.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

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
// SAXPY kernel (as in the original)
// -----------------------------------------------------------------------------
__global__
void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// -----------------------------------------------------------------------------
// CPU SAXPY (reference implementation)
// -----------------------------------------------------------------------------
void saxpy_cpu(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

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
// Main: Run SAXPY experiments over various vector sizes
// -----------------------------------------------------------------------------
int main(void) {
    printf("=== SAXPY Experiments ===\n");
    printf("CSV Format: VectorSize,GPUSaxpyTime_ms,CPUSaxpyTime_ms,MaxError\n");

    // Test vector sizes from 2^16 to 2^23 (65,536 to ~8 million elements)
    int sizes[] = {1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    float a = 2.0f;
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        float *h_x = (float*) malloc(N * sizeof(float));
        float *h_y = (float*) malloc(N * sizeof(float));
        float *h_y_cpu = (float*) malloc(N * sizeof(float));

        // Initialize input data
        for (int i = 0; i < N; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
            h_y_cpu[i] = 2.0f;
        }

        // Allocate device memory
        float *d_x = NULL, *d_y = NULL;
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

        // GPU SAXPY timing
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        cudaEvent_t start_gpu = get_time();
        saxpy<<<grid, block>>>(N, a, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t end_gpu = get_time();
        float gpuTime = elapsed_time(start_gpu, end_gpu);

        // Copy GPU result back to host
        CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

        // CPU SAXPY timing
        cudaEvent_t start_cpu = get_time();
        saxpy_cpu(N, a, h_x, h_y_cpu);
        cudaEvent_t end_cpu = get_time();
        float cpuTime = elapsed_time(start_cpu, end_cpu);

        // Calculate maximum error between GPU and CPU results
        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            float err = fabs(h_y[i] - h_y_cpu[i]);
            if (err > maxError)
                maxError = err;
        }
        printf("%d,%.4f,%.4f,%.5f\n", N, gpuTime, cpuTime, maxError);

        free(h_x);
        free(h_y);
        free(h_y_cpu);
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
