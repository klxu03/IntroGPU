/******************************************************************************
 * host_memory.cu
 *
 * Revised SAXPY experiments over a wider range of vector sizes.
 * Collects GPU and CPU timings (using CUDA events) and prints results in CSV.
 *
 * Build with: nvcc -O2 -o host_memory.exe host_memory.cu
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(err);                                         \
    }                                                    \
} while(0)

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

void saxpy_cpu(int n, float a, const float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

__host__ cudaEvent_t get_time(void) {
    cudaEvent_t t;
    cudaEventCreate(&t);
    cudaEventRecord(t, 0);
    cudaEventSynchronize(t);
    return t;
}

__host__ float elapsed_time(cudaEvent_t start, cudaEvent_t end) {
    float ms;
    cudaEventElapsedTime(&ms, start, end);
    return ms;
}

int main(void) {
    printf("=== SAXPY Experiments ===\n");
    printf("CSV Format: VectorSize,GPUSaxpyTime_ms,CPUSaxpyTime_ms,MaxError\n");

    // Test vector sizes from 2^16 up to 2^24
    int sizes[] = {1<<16, 1<<17, 1<<18, 1<<19, 1<<20, 1<<21, 1<<22, 1<<23, 1<<24};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < numSizes; s++) {
        int N = sizes[s];
        float a = 2.0f;
        float *h_x = (float*) malloc(N * sizeof(float));
        float *h_y = (float*) malloc(N * sizeof(float));
        float *h_y_cpu = (float*) malloc(N * sizeof(float));

        // Initialize input arrays
        for (int i = 0; i < N; i++) {
            h_x[i] = 1.0f;
            h_y[i] = 2.0f;
            h_y_cpu[i] = 2.0f;
        }

        float *d_x, *d_y;
        CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        // Time GPU SAXPY
        cudaEvent_t start_gpu = get_time();
        saxpy_kernel<<<grid, block>>>(N, a, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t end_gpu = get_time();
        float gpuTime = elapsed_time(start_gpu, end_gpu);

        CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Time CPU SAXPY
        cudaEvent_t start_cpu = get_time();
        saxpy_cpu(N, a, h_x, h_y_cpu);
        cudaEvent_t end_cpu = get_time();
        float cpuTime = elapsed_time(start_cpu, end_cpu);

        // Calculate maximum error
        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = fabs(h_y[i] - h_y_cpu[i]);
            if (diff > maxError)
                maxError = diff;
        }

        printf("%d,%.4f,%.4f,%.5f\n", N, gpuTime, cpuTime, maxError);

        free(h_x);
        free(h_y);
        free(h_y_cpu);
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
    }

    cudaDeviceReset();
    return 0;
}
