#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

// ------------------------------------------------------------------
// KERNELS
// ------------------------------------------------------------------

__global__ void addKernel(const int* arr1, const int* arr2,
                          int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = arr1[idx] + arr2[idx];
    }
}

__global__ void subtractKernel(const int* arr1, const int* arr2,
                               int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = arr1[idx] - arr2[idx];
    }
}

__global__ void multiplyKernel(const int* arr1, const int* arr2,
                               int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = arr1[idx] * arr2[idx];
    }
}

__global__ void modBranchKernel(const int* arr1, const int* arr2,
                                int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        if (arr2[idx] == 0) {
            out[idx] = arr1[idx] % 1;
        } else {
            out[idx] = arr1[idx] % arr2[idx];
        }
    }
}

__global__ void modNoBranchKernel(const int* arr1, const int* arr2,
                                  int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        int divisor = arr2[idx] + (arr2[idx] == 0);
        out[idx] = arr1[idx] % divisor;
    }
}

__global__ void modNoBranchSeparateKernel(const int* arr1, int* out,
                                          int totalThreads, int fixedDivisor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = arr1[idx] % fixedDivisor;
    }
}

__global__ void modBy1Kernel(const int* arr1, int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = 0;
    }
}

__global__ void modBy2Kernel(const int* arr1, int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = arr1[idx] & 1;
    }
}

__global__ void modBy3Kernel(const int* arr1, int* out, int totalThreads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalThreads) {
        out[idx] = arr1[idx] % 3;
    }
}

// ------------------------------------------------------------------
// FUNCTIONS FOR CODE ORGANIZATION
// ------------------------------------------------------------------

void parseArgs(int argc, char** argv, int &totalThreads, int &blockSize)
{
    totalThreads = (1 << 20);
    blockSize    = 256;

    if (argc >= 2) totalThreads = atoi(argv[1]);
    if (argc >= 3) blockSize = atoi(argv[2]);
}

void setupGridDims(int &totalThreads, int &blockSize,
                   int &numBlocks)
{
    numBlocks = totalThreads / blockSize;
    if (totalThreads % blockSize != 0) {
        numBlocks++;
        totalThreads = numBlocks * blockSize;
        printf("Warning: Threads not divisible by blockSize. "
               "Rounded up to %d.\n", totalThreads);
    }
    printf("Running with %d total threads (%d blocks of %d)\n",
           totalThreads, numBlocks, blockSize);
}

void fillArrays(int* arr1, int* arr2, int totalThreads)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < totalThreads; i++) {
        arr1[i] = i;
        arr2[i] = rand() % 4;
    }
}

void partitionArrays(const int* h_arr1, const int* h_arr2,
                     int totalThreads, int &count0, int &count1,
                     int &count2, int &count3, int* &h_p0, int* &h_p1,
                     int* &h_p2, int* &h_p3)
{
    count0 = count1 = count2 = count3 = 0;
    for (int i = 0; i < totalThreads; i++) {
        switch(h_arr2[i]) {
            case 0: count0++; break;
            case 1: count1++; break;
            case 2: count2++; break;
            case 3: count3++; break;
        }
    }
    h_p0 = (int*)malloc(count0 * sizeof(int));
    h_p1 = (int*)malloc(count1 * sizeof(int));
    h_p2 = (int*)malloc(count2 * sizeof(int));
    h_p3 = (int*)malloc(count3 * sizeof(int));
    int i0=0, i1=0, i2=0, i3=0;
    for (int i = 0; i < totalThreads; i++) {
        switch(h_arr2[i]) {
            case 0: h_p0[i0++] = h_arr1[i]; break;
            case 1: h_p1[i1++] = h_arr1[i]; break;
            case 2: h_p2[i2++] = h_arr1[i]; break;
            case 3: h_p3[i3++] = h_arr1[i]; break;
        }
    }
}

void showSample(const int* h_arr1, const int* h_arr2,
                const int* h_add, const int* h_sub,
                const int* h_mul, const int* h_mod1)
{
    printf("\nSample results (first 5):\n");
    for (int i = 0; i < 5; i++) {
        printf("i=%d | arr1=%d arr2=%d | add=%d sub=%d "
               "mul=%d | modB=%d\n",
               i, h_arr1[i], h_arr2[i], h_add[i],
               h_sub[i], h_mul[i], h_mod1[i]);
    }
}

void timeKernels(int numBlocks, int blockSize, int totalThreads,
                 int* d_arr1, int* d_arr2, int* d_modOut1, int* d_modOut2,
                 int* h_arr1, int* h_arr2)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float msBranch=0.f, msNoBranch=0.f, msSep=0.f;

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        modBranchKernel<<<numBlocks, blockSize>>>(d_arr1,
          d_arr2, d_modOut1, totalThreads);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msBranch, start, stop);

    int c0, c1, c2, c3;
    int *h_p0, *h_p1, *h_p2, *h_p3;
    partitionArrays(h_arr1, h_arr2, totalThreads, c0, c1, c2, c3,
                    h_p0, h_p1, h_p2, h_p3);

    int *d_p0, *d_p1, *d_p2, *d_p3;
    int *d_o0, *d_o1, *d_o2, *d_o3;

    cudaMalloc((void**)&d_p0, c0*sizeof(int));
    cudaMalloc((void**)&d_p1, c1*sizeof(int));
    cudaMalloc((void**)&d_p2, c2*sizeof(int));
    cudaMalloc((void**)&d_p3, c3*sizeof(int));
    cudaMalloc((void**)&d_o0, c0*sizeof(int));
    cudaMalloc((void**)&d_o1, c1*sizeof(int));
    cudaMalloc((void**)&d_o2, c2*sizeof(int));
    cudaMalloc((void**)&d_o3, c3*sizeof(int));

    cudaMemcpy(d_p0, h_p0, c0*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p1, h_p1, c1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p2, h_p2, c2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p3, h_p3, c3*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        modNoBranchKernel<<<numBlocks, blockSize>>>(d_arr1,
          d_arr2, d_modOut2, totalThreads);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msNoBranch, start, stop);

    cudaEventRecord(start);
    for(int i=0; i<10; i++) {
        if(c0>0)
            modBy1Kernel<<<(c0+blockSize-1)/blockSize, blockSize>>>
                (d_p0, d_o0, c0);
        if(c1>0)
            modBy1Kernel<<<(c1+blockSize-1)/blockSize, blockSize>>>
                (d_p1, d_o1, c1);
        if(c2>0)
            modBy2Kernel<<<(c2+blockSize-1)/blockSize, blockSize>>>
                (d_p2, d_o2, c2);
        if(c3>0)
            modBy3Kernel<<<(c3+blockSize-1)/blockSize, blockSize>>>
                (d_p3, d_o3, c3);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msSep, start, stop);

    printf("\nPerformance (10 iterations each):\n");
    printf("1) Branching mod:    %.3f ms\n", msBranch);
    printf("2) No-branch mod:    %.3f ms\n", msNoBranch);
    printf("3) Partitioned mod:  %.3f ms\n", msSep);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_p0);
    free(h_p1);
    free(h_p2);
    free(h_p3);

    cudaFree(d_p0);
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_p3);
    cudaFree(d_o0);
    cudaFree(d_o1);
    cudaFree(d_o2);
    cudaFree(d_o3);
}

// ------------------------------------------------------------------
// MAIN
// ------------------------------------------------------------------

int main(int argc, char** argv)
{
    int totalThreads, blockSize, numBlocks;
    parseArgs(argc, argv, totalThreads, blockSize);
    setupGridDims(totalThreads, blockSize, numBlocks);

    size_t bytes = totalThreads * sizeof(int);
    int* h_arr1   = (int*)malloc(bytes);
    int* h_arr2   = (int*)malloc(bytes);
    int* h_addOut = (int*)malloc(bytes);
    int* h_subOut = (int*)malloc(bytes);
    int* h_mulOut = (int*)malloc(bytes);
    int* h_mod1   = (int*)malloc(bytes);

    fillArrays(h_arr1, h_arr2, totalThreads);

    int *d_arr1, *d_arr2;
    int *d_addOut, *d_subOut, *d_mulOut;
    int *d_modOut1, *d_modOut2;

    cudaMalloc((void**)&d_arr1, bytes);
    cudaMalloc((void**)&d_arr2, bytes);
    cudaMalloc((void**)&d_addOut, bytes);
    cudaMalloc((void**)&d_subOut, bytes);
    cudaMalloc((void**)&d_mulOut, bytes);
    cudaMalloc((void**)&d_modOut1, bytes);
    cudaMalloc((void**)&d_modOut2, bytes);

    cudaMemcpy(d_arr1, h_arr1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, h_arr2, bytes, cudaMemcpyHostToDevice);

    addKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2,
                                        d_addOut, totalThreads);
    subtractKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2,
                                             d_subOut, totalThreads);
    multiplyKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2,
                                             d_mulOut, totalThreads);
    modBranchKernel<<<numBlocks, blockSize>>>(d_arr1, d_arr2,
                                              d_modOut1, totalThreads);

    cudaDeviceSynchronize();

    cudaMemcpy(h_addOut, d_addOut, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_subOut, d_subOut, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mulOut, d_mulOut, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mod1,   d_modOut1, bytes, cudaMemcpyDeviceToHost);

    showSample(h_arr1, h_arr2, h_addOut, h_subOut,
               h_mulOut, h_mod1);

    timeKernels(numBlocks, blockSize, totalThreads, d_arr1, d_arr2,
                d_modOut1, d_modOut2, h_arr1, h_arr2);

    free(h_arr1);
    free(h_arr2);
    free(h_addOut);
    free(h_subOut);
    free(h_mulOut);
    free(h_mod1);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_addOut);
    cudaFree(d_subOut);
    cudaFree(d_mulOut);
    cudaFree(d_modOut1);
    cudaFree(d_modOut2);

    cudaDeviceReset();
    return 0;
}