#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: unchanged from your original
__global__ void what_is_my_id_2d_A(
    unsigned int * const block_x,
    unsigned int * const block_y,
    unsigned int * const thread,
    unsigned int * const calc_thread,
    unsigned int * const x_thread,
    unsigned int * const y_thread,
    unsigned int * const grid_dimx,
    unsigned int * const block_dimx,
    unsigned int * const grid_dimy,
    unsigned int * const block_dimy)
{
    const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int thread_idx = ((gridDim.x * blockDim.x) * idy) + idx;

    block_x[thread_idx]   = blockIdx.x;
    block_y[thread_idx]   = blockIdx.y;
    thread[thread_idx]    = threadIdx.x;
    calc_thread[thread_idx] = thread_idx;
    x_thread[thread_idx]  = idx;
    y_thread[thread_idx]  = idy;
    grid_dimx[thread_idx] = gridDim.x;
    block_dimx[thread_idx]= blockDim.x;
    grid_dimy[thread_idx] = gridDim.y;
    block_dimy[thread_idx]= blockDim.y;
}

// Increase the problem size for timing tests
#define ARRAY_SIZE_X 256
#define ARRAY_SIZE_Y 256
#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(unsigned int)))

// Declare CPU arrays (for a large problem, consider dynamic allocation)
unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
unsigned int cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];

int main(void)
{
    // --- Define different thread/block configurations ---
    // Configuration A:
    const dim3 threads_rect(32, 4);
    const dim3 blocks_rect(1, 4);
    
    // Configuration B:
    const dim3 threads_square(16, 8);
    const dim3 blocks_square(2, 2);
    
    // Configuration C: (Optimal occupancy test for a 256x256 grid)
    // Total threads = 256x256, so blocks = (256/16, 256/16) = (16,16) with threads per block (16,16)
    const dim3 threads_16x16(16, 16);
    const dim3 blocks_16x16(ARRAY_SIZE_X / threads_16x16.x, ARRAY_SIZE_Y / threads_16x16.y);

    // --- Create CUDA events for timing ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    unsigned int * gpu_block_x, * gpu_block_y, * gpu_thread, * gpu_calc_thread;
    unsigned int * gpu_xthread, * gpu_ythread, * gpu_grid_dimx, * gpu_block_dimx;
    unsigned int * gpu_grid_dimy, * gpu_block_dimy;
    
    cudaMalloc((void **)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_thread,  ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_xthread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_ythread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void **)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);

    // Loop over kernel configurations
    for(int kernel = 0; kernel < 3; kernel++)
    {
        // Record start time
        cudaEventRecord(start, 0);
        
        switch(kernel)
        {
            case 0:
                // Configuration A: threads_rect / blocks_rect
                what_is_my_id_2d_A<<<blocks_rect, threads_rect>>>(gpu_block_x, gpu_block_y,
                      gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread,
                      gpu_grid_dimx, gpu_block_dimx, gpu_grid_dimy, gpu_block_dimy);
                break;
            case 1:
                // Configuration B: threads_square / blocks_square
                what_is_my_id_2d_A<<<blocks_square, threads_square>>>(gpu_block_x, gpu_block_y,
                      gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread,
                      gpu_grid_dimx, gpu_block_dimx, gpu_grid_dimy, gpu_block_dimy);
                break;
            case 2:
                // Configuration C: threads_16x16 / blocks_16x16
                what_is_my_id_2d_A<<<blocks_16x16, threads_16x16>>>(gpu_block_x, gpu_block_y,
                      gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread,
                      gpu_grid_dimx, gpu_block_dimx, gpu_grid_dimy, gpu_block_dimy);
                break;
            default:
                exit(1);
        }
        
        // Record stop time and synchronize
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("\nKernel %d execution time: %f ms\n", kernel, elapsedTime);
        
        // Copy results back to CPU (if needed for validation/printing)
        cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xthread, gpu_xthread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_ythread, gpu_ythread, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES, cudaMemcpyDeviceToHost);
        
        // (Optional) Print a few entries to validate kernel execution
        printf("First element details: CT: %2u BKX: %1u BKY: %1u TID: %2u XTID: %2u YTID: %2u\n",
               cpu_calc_thread[0][0], cpu_block_x[0][0], cpu_block_y[0][0],
               cpu_thread[0][0], cpu_xthread[0][0], cpu_ythread[0][0]);
    }
    
    // Clean up GPU memory and events
    cudaFree(gpu_block_x);
    cudaFree(gpu_block_y);
    cudaFree(gpu_thread);
    cudaFree(gpu_calc_thread);
    cudaFree(gpu_xthread);
    cudaFree(gpu_ythread);
    cudaFree(gpu_grid_dimx);
    cudaFree(gpu_block_dimx);
    cudaFree(gpu_grid_dimy);
    cudaFree(gpu_block_dimy);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}