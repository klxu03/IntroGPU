#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { cudaError_t err = call; if(err != cudaSuccess){ std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in file " << __FILE__ << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)
#define CHECK_CUFFT(call) do { cufftResult err = call; if(err != CUFFT_SUCCESS){ std::cerr << "CUFFT error: " << err << " in file " << __FILE__ << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)
#define CHECK_CUDNN(call) do { cudnnStatus_t status = call; if(status != CUDNN_STATUS_SUCCESS){ std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " in file " << __FILE__ << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t status = call; if(status != CUBLAS_STATUS_SUCCESS){ std::cerr << "cuBLAS error: " << status << " in file " << __FILE__ << " line " << __LINE__ << std::endl; exit(EXIT_FAILURE); } } while(0)

const int NX = 256;
const int NY = 256;
const int INPUT_SIZE = NX * NY;
const int OUTPUT_SIZE = 1024;

void printStatsFloat(const float* data, int n, const char* label) {
    float minVal = data[0], maxVal = data[0], sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
        sum += data[i];
    }
    float mean = sum / n;
    std::cout << label << " - min: " << minVal << " max: " << maxVal << " mean: " << mean << std::endl;
}

void createSyntheticImage(cufftComplex *h_image, int n) {
    for (int i = 0; i < n; ++i) {
        h_image[i].x = static_cast<float>(rand() % 256);
        h_image[i].y = 0.0f;
    }
}

__global__ void frequencyFilter(cufftComplex *data, int nx, int ny, float threshold, float scaleFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx < total) {
        float mag = sqrtf(data[idx].x * data[idx].x + data[idx].y * data[idx].y);
        if (mag < threshold) {
            data[idx].x *= scaleFactor;
            data[idx].y *= scaleFactor;
        }
    }
}

__global__ void scaleOutput(cufftComplex *data, int total, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

void processFFT(cufftHandle fftPlan, cufftComplex* d_image, int nx, int ny, int threadsPerBlock, int blocks,
                float threshold, float scaleFactor, float*& h_enhanced) {
    CHECK_CUFFT(cufftExecC2C(fftPlan, d_image, d_image, CUFFT_FORWARD));
    frequencyFilter<<<blocks, threadsPerBlock>>>(d_image, nx, ny, threshold, scaleFactor);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUFFT(cufftExecC2C(fftPlan, d_image, d_image, CUFFT_INVERSE));
    int totalElements = nx * ny;
    cufftComplex* h_temp = new cufftComplex[totalElements];
    CHECK_CUDA(cudaMemcpy(h_temp, d_image, totalElements * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    for (int i = 0; i < totalElements; i++) {
        h_temp[i].x /= (nx * ny);
        h_temp[i].y /= (nx * ny);
    }
    h_enhanced = new float[totalElements];
    for (int i = 0; i < totalElements; i++) {
        h_enhanced[i] = h_temp[i].x;
    }
    delete[] h_temp;
}

void fcLayer(cublasHandle_t cublasHandle, float* d_fc_input, float* d_fc_output, int input_size, int output_size,
             float* h_W, float* h_b, float* h_fc_output) {
    float *d_W, *d_b;
    CHECK_CUDA(cudaMalloc((void**)&d_W, output_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, output_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, output_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, output_size * sizeof(float), cudaMemcpyHostToDevice));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, output_size, 1, input_size, &alpha,
                              d_W, output_size, d_fc_input, input_size, &beta, d_fc_output, output_size));
    for (int i = 0; i < output_size; ++i)
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, 1, &alpha, d_b + i, 1, d_fc_output + i, 1));
    CHECK_CUDA(cudaMemcpy(h_fc_output, d_fc_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_b));
}

void reluActivation(cudnnHandle_t cudnnHandle, float* d_fc_output, int output_size, float* d_fc_relu, float* h_fc_relu) {
    cudnnTensorDescriptor_t tensorDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, output_size, 1));
    cudnnActivationDescriptor_t actDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&actDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnActivationForward(cudnnHandle, actDesc, &alpha, tensorDesc, d_fc_output, &beta, tensorDesc, d_fc_relu));
    CHECK_CUDA(cudaMemcpy(h_fc_relu, d_fc_relu, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(actDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
}

int main() {
    cufftComplex* h_image = new cufftComplex[INPUT_SIZE];
    createSyntheticImage(h_image, INPUT_SIZE);
    float *temp = new float[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) temp[i] = h_image[i].x;
    printStatsFloat(temp, INPUT_SIZE, "Stage 1 - Synthetic Image");
    delete[] temp;
    
    cufftComplex* d_image;
    CHECK_CUDA(cudaMalloc((void**)&d_image, INPUT_SIZE * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_image, h_image, INPUT_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    cufftHandle fftPlan;
    CHECK_CUFFT(cufftPlan2d(&fftPlan, NX, NY, CUFFT_C2C));
    float* h_enhanced = nullptr;
    int threadsPerBlock = 256;
    int blocks = (INPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    processFFT(fftPlan, d_image, NX, NY, threadsPerBlock, blocks, 100.0f, 0.5f, h_enhanced);
    printStatsFloat(h_enhanced, INPUT_SIZE, "Stage 2 - After Inverse FFT (Normalized on Host)");
    delete[] h_image;

    float* d_fc_input;
    CHECK_CUDA(cudaMalloc((void**)&d_fc_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_fc_input, h_enhanced, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    float* d_fc_output;
    CHECK_CUDA(cudaMalloc((void**)&d_fc_output, OUTPUT_SIZE * sizeof(float)));
    float* h_W = new float[OUTPUT_SIZE * INPUT_SIZE];
    float* h_b = new float[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE * INPUT_SIZE; i++)
        h_W[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        h_b[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    float* h_fc_output = new float[OUTPUT_SIZE];
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    fcLayer(cublasHandle, d_fc_input, d_fc_output, INPUT_SIZE, OUTPUT_SIZE, h_W, h_b, h_fc_output);
    printStatsFloat(h_fc_output, OUTPUT_SIZE, "Stage 3 - FC Raw Output");

    cudnnHandle_t cudnnHandle;
    CHECK_CUDNN(cudnnCreate(&cudnnHandle));
    float* d_fc_relu;
    CHECK_CUDA(cudaMalloc((void**)&d_fc_relu, OUTPUT_SIZE * sizeof(float)));
    float* h_fc_relu = new float[OUTPUT_SIZE];
    reluActivation(cudnnHandle, d_fc_output, OUTPUT_SIZE, d_fc_relu, h_fc_relu);
    printStatsFloat(h_fc_relu, OUTPUT_SIZE, "Stage 4 - After ReLU");
    
    std::cout << "Raw FC Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << h_fc_output[i] << " ";
    std::cout << std::endl;
    std::cout << "ReLU FC Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << h_fc_relu[i] << " ";
    std::cout << std::endl;
    
    delete[] h_enhanced;
    delete[] h_W;
    delete[] h_b;
    delete[] h_fc_output;
    delete[] h_fc_relu;
    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaFree(d_fc_input));
    CHECK_CUDA(cudaFree(d_fc_output));
    CHECK_CUDA(cudaFree(d_fc_relu));
    CHECK_CUFFT(cufftDestroy(fftPlan));
    CHECK_CUDNN(cudnnDestroy(cudnnHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));
    return 0;
}
