/******************************************************************************
 * host_memory_experiments.cu
 *
 * Demonstrates a SAXPY operation with multiple vector sizes. Collects kernel
 * timing and optional CPU timing, printing out results in CSV format.
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
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(err);                                                 \
    }                                                              \
} while(0)

// -----------------------------------------------------------------------------
// SAXPY kernel
// -----------------------------------------------------------------------------
__global__ void saxpy_kernel(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

// -----------------------------------------------------------------------------
// CPU SAXPY (optional reference)
// -----------------------------------------------------------------------------
void saxpy_cpu(int n, float a, const float *x, float *y)
{
    for (int i = 0; i < n; i++) {
        y[i] = a*x[i] + y[i];
    }
}

// -----------------------------------------------------------------------------
// Time helper
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
// Main
// -----------------------------------------------------------------------------
int main(void)
{
    // We'll test vector sizes from 2^16 to 2^23, for example
    // (65,536 up to ~8 million). Adjust as you like.
    int sizes[] = {1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23};

    // Print CSV header
    printf("VectorSize,GPUSaxpyTime(ms),CPUSaxpyTime(ms),MaxError\n");

    for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
        int N = sizes[s];
        float a = 2.0f;

        // Allocate host arrays
        float *h_x = (float*) malloc(N * sizeof(float));
        float *h_y = (float*) malloc(N * sizeof(float));
        float *h_y_cpu = (float*) malloc(N * sizeof(float)); // separate copy for CPU test

        // Initialize data
        for (int i = 0; i < N; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
            h_y_cpu[i] = 2.0f;  // same initial data for CPU test
        }

        // Allocate device arrays
        float *d_x = nullptr, *d_y = nullptr;
        CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice));

        // ----------------------------
        // Time the GPU SAXPY
        // ----------------------------
        dim3 block(256);
        dim3 grid((N + block.x - 1)/block.x);

        cudaEvent_t start_gpu = get_time();
        saxpy_kernel<<<grid, block>>>(N, a, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t end_gpu = get_time();

        float gpuTime = elapsed_time(start_gpu, end_gpu);

        // Copy back
        CUDA_CHECK(cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));

        // ----------------------------
        // Time the CPU SAXPY
        // ----------------------------
        cudaEvent_t start_cpu = get_time();
        saxpy_cpu(N, a, h_x, h_y_cpu);
        cudaEvent_t end_cpu = get_time();

        float cpuTime = elapsed_time(start_cpu, end_cpu);

        // ----------------------------
        // Check max error
        // ----------------------------
        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = fabs(h_y[i] - h_y_cpu[i]);
            if (diff > maxError) {
                maxError = diff;
            }
        }

        // Print results in CSV
        printf("%d,%.4f,%.4f,%.5f\n", N, gpuTime, cpuTime, maxError);

        // Cleanup
        free(h_x);
        free(h_y);
        free(h_y_cpu);
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    cudaDeviceReset();
    return 0;
}
