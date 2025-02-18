/******************************************************************************
 * host_memory_experiments.cu
 *
 * Demonstrates SAXPY on various vector sizes and measures kernel execution
 * times on the GPU (using CUDA events) and the CPU (using clock()).
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",       \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(err);                                                 \
    }                                                              \
} while(0)

__global__ void saxpy_kernel(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void) {
    printf("=== SAXPY Experiments ===\n");
    printf("CSV Format: VectorSize,GPUSaxpyTime_ms,CPUSaxpyTime_ms,MaxError\n");

    int sizes[] = {65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216};
    float a = 2.0f;
    for (int s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
        int N = sizes[s];
        float *h_x = (float*) malloc(N * sizeof(float));
        float *h_y = (float*) malloc(N * sizeof(float));
        float *h_y_cpu = (float*) malloc(N * sizeof(float));
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
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        cudaEventRecord(start_gpu, 0);
        saxpy_kernel<<<grid, block>>>(N, a, d_x, d_y);
        cudaEventRecord(stop_gpu, 0);
        cudaEventSynchronize(stop_gpu);
        float gpuTime;
        cudaEventElapsedTime(&gpuTime, start_gpu, stop_gpu);
        CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
        cudaEventDestroy(start_gpu);
        cudaEventDestroy(stop_gpu);

        clock_t start_cpu = clock();
        for (int i = 0; i < N; i++) {
            h_y_cpu[i] = a * h_x[i] + h_y_cpu[i];
        }
        clock_t end_cpu = clock();
        double cpuTime = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0;

        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            float diff = fabs(h_y[i] - h_y_cpu[i]);
            if (diff > maxError)
                maxError = diff;
        }
        printf("%d,%.4f,%.4f,%.5f\n", N, gpuTime, cpuTime, maxError);

        free(h_x); free(h_y); free(h_y_cpu);
        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    }
    cudaDeviceReset();
    return 0;
}
