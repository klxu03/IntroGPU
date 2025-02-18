/* *
 * Modified SAXPY sample for improved timing output and experiment logging.
 * Excessive per-element printing has been removed. Instead, a loop of experiments
 * on varying vector sizes is run, and key timings (kernel time and max error) are logged.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// SAXPY kernel from NVIDIA samples.
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a*x[i] + y[i];
}

int main(void)
{
    // Define a few experiment sizes (power-of-two sizes)
    const int num_experiments = 4;
    int sizes[num_experiments] = {1 << 18, 1 << 19, 1 << 20, 1 << 21};

    // Log CSV header: Experiment, VectorSize, KernelTime(ms), MaxError
    printf("Experiment,VectorSize,KernelTime(ms),MaxError\n");

    for (int exp = 0; exp < num_experiments; exp++){
        int N = sizes[exp];
        float *x, *y, *d_x, *d_y;
        size_t size_in_bytes = N * sizeof(float);

        // Allocate host memory
        x = (float*)malloc(size_in_bytes);
        y = (float*)malloc(size_in_bytes);

        // Allocate device memory
        cudaMalloc(&d_x, size_in_bytes); 
        cudaMalloc(&d_y, size_in_bytes);

        // Initialize host arrays
        for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        // Copy data to device
        cudaMemcpy(d_x, x, size_in_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, size_in_bytes, cudaMemcpyHostToDevice);

        // Setup timing events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Launch SAXPY: use enough threads to cover all elements.
        saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float kernelTime = 0.0f;
        cudaEventElapsedTime(&kernelTime, start, stop);

        // Copy result back to host
        cudaMemcpy(y, d_y, size_in_bytes, cudaMemcpyDeviceToHost);

        // Verify the result: expected value is 2 + 2*1 = 4 for every element.
        float maxError = 0.0f;
        for (int i = 0; i < N; i++){
            float error = fabs(y[i] - 4.0f);
            if(error > maxError)
                maxError = error;
        }

        // Log experiment results in CSV format.
        printf("%d,%d,%.3f,%.5f\n", exp, N, kernelTime, maxError);

        // Cleanup events and memory
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_x);
        cudaFree(d_y);
        free(x);
        free(y);
    }

    return 0;
}
