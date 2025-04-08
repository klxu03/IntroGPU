#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cudnn.h>
#include <cublas_v2.h>

// --- Error Checking Macros ---
#define CHECK_CUDA(call) do {                                \
    cudaError_t err = call;                                  \
    if(err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__                  \
                  << " line " << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

#define CHECK_CUFFT(call) do {                                \
    cufftResult err = call;                                  \
    if(err != CUFFT_SUCCESS) {                               \
        std::cerr << "CUFFT error: " << err                 \
                  << " in file " << __FILE__                 \
                  << " line " << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

#define CHECK_CUDNN(call) do {                                \
    cudnnStatus_t status = call;                             \
    if(status != CUDNN_STATUS_SUCCESS) {                     \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) \
                  << " in file " << __FILE__                 \
                  << " line " << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

#define CHECK_CUBLAS(call) do {                              \
    cublasStatus_t status = call;                            \
    if(status != CUBLAS_STATUS_SUCCESS) {                    \
        std::cerr << "cuBLAS error: " << status             \
                  << " in file " << __FILE__                 \
                  << " line " << __LINE__ << std::endl;      \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
} while(0)

// --- Global Constants ---
const int NX = 256;   // Image width
const int NY = 256;   // Image height
const int INPUT_SIZE = NX * NY;    // 65536 elements
const int OUTPUT_SIZE = 1024;      // FC layer output size

// --- CUDA Kernel: Frequency Filter ---
// This kernel goes through the frequency domain data (complex numbers) and
// if the magnitude of a frequency coefficient is below a threshold, scales it.
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

// --- CUDA Kernel: Scale Inverse FFT Output ---
// Scales the output of inverse FFT by a given factor (normalization).
__global__ void scaleOutput(cufftComplex *data, int total, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < total) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

int main() {
    // -------------------------------
    // 1. Create a Synthetic Image
    // -------------------------------
    // Create a grayscale image with random pixel intensities (0-255)
    cufftComplex *h_image = new cufftComplex[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; ++i) {
        h_image[i].x = static_cast<float>(rand() % 256); // Pixel intensity
        h_image[i].y = 0.0f;  // Imaginary part is 0 since image is real
    }

    // -------------------------------
    // 2. Copy Image to Device Memory
    // -------------------------------
    cufftComplex *d_image;
    CHECK_CUDA(cudaMalloc((void**)&d_image, INPUT_SIZE * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMemcpy(d_image, h_image, INPUT_SIZE * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    // -------------------------------
    // 3. Forward FFT using cuFFT
    // -------------------------------
    cufftHandle fftPlan;
    CHECK_CUFFT(cufftPlan2d(&fftPlan, NX, NY, CUFFT_C2C));
    CHECK_CUFFT(cufftExecC2C(fftPlan, d_image, d_image, CUFFT_FORWARD));

    // -------------------------------
    // 4. Apply Frequency Filtering in the Frequency Domain
    // -------------------------------
    int totalElements = INPUT_SIZE;
    int threadsPerBlock = 256;
    int blocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    float threshold = 100.0f;     // Example threshold for frequency magnitude
    float scaleFactor = 0.5f;     // Scale frequency component if below threshold
    frequencyFilter<<<blocks, threadsPerBlock>>>(d_image, NX, NY, threshold, scaleFactor);
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------------------------------
    // 5. Inverse FFT to Return to Spatial Domain
    // -------------------------------
    CHECK_CUFFT(cufftExecC2C(fftPlan, d_image, d_image, CUFFT_INVERSE));
    // Normalize the result by dividing by the total number of elements (NX*NY)
    scaleOutput<<<blocks, threadsPerBlock>>>(d_image, totalElements, 1.0f / (NX * NY));
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------------------------------
    // 6. Retrieve Enhanced Image from Device
    // -------------------------------
    cufftComplex *h_enhanced = new cufftComplex[INPUT_SIZE];
    CHECK_CUDA(cudaMemcpy(h_enhanced, d_image, INPUT_SIZE * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

    // Prepare a float array from the real parts only (flattened image)
    float *h_float = new float[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; ++i) {
        h_float[i] = h_enhanced[i].x;
    }

    // -------------------------------
    // 7. Fully Connected (FC) Layer (MLP) using cuBLAS
    // -------------------------------
    // Here we treat the enhanced image as a single vector (of 65536 pixels)
    // and pass it through a fully connected layer with randomly initialized weights and biases.
    // This simulates a normal neural network layer.
    
    // Allocate device memory for the input vector (as a column vector)
    float *d_fc_input;
    CHECK_CUDA(cudaMalloc((void**)&d_fc_input, INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_fc_input, h_float, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate memory for FC output (size OUTPUT_SIZE x 1)
    float *d_fc_output;
    CHECK_CUDA(cudaMalloc((void**)&d_fc_output, OUTPUT_SIZE * sizeof(float)));

    // Allocate and initialize random weights and biases on host
    float *h_W = new float[OUTPUT_SIZE * INPUT_SIZE]; // Weight matrix of shape (OUTPUT_SIZE x INPUT_SIZE)
    float *h_b = new float[OUTPUT_SIZE]; // Bias vector for OUTPUT_SIZE neurons
    for (int i = 0; i < OUTPUT_SIZE * INPUT_SIZE; ++i) {
        // Random weights in range [-0.1, 0.1]
        h_W[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        // Random biases in range [-0.1, 0.1]
        h_b[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }

    // Allocate device memory for weights and biases
    float *d_W, *d_b;
    CHECK_CUDA(cudaMalloc((void**)&d_W, OUTPUT_SIZE * INPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, OUTPUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_W, h_W, OUTPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // Perform matrix multiplication: FC_output = W * x
    // Here, x is d_fc_input (shape: INPUT_SIZE x 1) and W is of shape (OUTPUT_SIZE x INPUT_SIZE)
    // We'll use cublasSgemm; note that cuBLAS uses column-major ordering.
    // To keep it simple, we pretend our data is in column-major order.
    float alpha_cublas = 1.0f, beta_cublas = 0.0f;
    // We set: m = OUTPUT_SIZE, n = 1, k = INPUT_SIZE
    CHECK_CUBLAS(cublasSgemm(
        cublasHandle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        OUTPUT_SIZE, 1, INPUT_SIZE,
        &alpha_cublas,
        d_W, OUTPUT_SIZE,
        d_fc_input, INPUT_SIZE,
        &beta_cublas,
        d_fc_output, OUTPUT_SIZE));

    // Add the bias vector: FC_output = FC_output + b
    // Use cublasSaxpy for vector addition: y = a*x + y. Here, a=1.
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, 1, &alpha_cublas, d_b + i, 1, d_fc_output + i, 1));
    }

    // Copy FC layer raw output back to host to see some negative values
    float *h_fc_output = new float[OUTPUT_SIZE];
    CHECK_CUDA(cudaMemcpy(h_fc_output, d_fc_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "Raw Fully-Connected Layer Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_fc_output[i] << " ";
    }
    std::cout << std::endl;

    // -------------------------------
    // 8. Optionally, Apply cuDNN Activation to FC Output
    // -------------------------------
    // Now we pass the FC output through a ReLU activation layer.
    // This will set negative values to 0.
    
    // For cuDNN, we need to reinterpret the FC output as a 4D tensor.
    // We'll treat it as a tensor of shape: (batch=1, channels=1, height=OUTPUT_SIZE, width=1)
    cudnnTensorDescriptor_t fcTensorDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&fcTensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(fcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, 1, OUTPUT_SIZE, 1));

    cudnnActivationDescriptor_t fcActDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&fcActDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(fcActDesc, CUDNN_ACTIVATION_RELU,
                                              CUDNN_PROPAGATE_NAN, 0.0));

    float *d_fc_relu;
    CHECK_CUDA(cudaMalloc((void**)&d_fc_relu, OUTPUT_SIZE * sizeof(float)));

    float alpha_relu = 1.0f, beta_relu = 0.0f;
    CHECK_CUDNN(cudnnActivationForward(cudnnHandle, fcActDesc, &alpha_relu,
                                        fcTensorDesc, d_fc_output,
                                        &beta_relu,
                                        fcTensorDesc, d_fc_relu));
    
    // Copy the ReLU output back to host
    float *h_fc_relu = new float[OUTPUT_SIZE];
    CHECK_CUDA(cudaMemcpy(h_fc_relu, d_fc_relu, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "ReLU-activated FC Layer Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << h_fc_relu[i] << " ";
    }
    std::cout << std::endl;

    // -------------------------------
    // 9. Cleanup: Free Memory and Destroy Handles
    // -------------------------------
    delete[] h_image;
    delete[] h_enhanced;
    delete[] h_float;
    delete[] h_W;
    delete[] h_b;
    delete[] h_fc_output;
    delete[] h_fc_relu;
    
    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaFree(d_fc_input));
    CHECK_CUDA(cudaFree(d_fc_output));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_fc_relu));
    
    CHECK_CUFFT(cufftDestroy(fftPlan));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(fcActDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(fcTensorDesc));
    CHECK_CUDNN(cudnnDestroy(cudnnHandle));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}

