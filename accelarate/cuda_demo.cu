#include <iostream>
#include <ctime>

__global__ void vectorAddKernel(float *pInVector, float *pOutVector, float *vec_2, float *vec_1, int len)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < len)
    {
        float nega = vec_2[index % 16];
        float pos = vec_1[index % 16];

        // Perform vector operations
        float in_val = pInVector[index];
        float out_val = (in_val * pos - in_val * nega);
        pOutVector[index] = out_val;
    }
}

int main(int argc, char const *argv[])
{
    int len = 2100000000;
    size_t size = len * sizeof(float);

    float *pInVector = new float[len];
    float *pOutVector = new float[len];

    // Initialize input data
    for (int i = 0; i < len; ++i)
    {
        pInVector[i] = 1.0f;
    }

    // Initialize vectors with values
    float vec_1[16] = {1, 1, 1, 1,
                       1, 1, 1, 1,
                       1, 1, 1, 1,
                       1, 1, 1, 1};
    float vec_2[16] = {-1, -1, -1, -1,
                       -1, -1, -1, -1,
                       -1, -1, -1, -1,
                       -1, -1, -1, -1};

    // Allocate device memory
    float *d_pInVector, *d_pOutVector, *d_vec_2, *d_vec_1;
    cudaMalloc(&d_pInVector, size);
    cudaMalloc(&d_pOutVector, size);
    cudaMalloc(&d_vec_2, 16 * sizeof(float));
    cudaMalloc(&d_vec_1, 16 * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_pInVector, pInVector, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_2, vec_2, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_1, vec_1, 16 * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel configuration
    int blockSize = 256;
    int numBlocks = (len + blockSize - 1) / blockSize;

    // Launch kernel
    std::clock_t start = std::clock();
    vectorAddKernel<<<numBlocks, blockSize>>>(d_pInVector, d_pOutVector, d_vec_2, d_vec_1, len);
    std::clock_t end = std::clock();
    double elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed calc time: " << elapsed << " seconds\n";

    // Copy results back to host
    cudaMemcpy(pOutVector, d_pOutVector, size, cudaMemcpyDeviceToHost);
    end = std::clock();
    elapsed = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Elapsed res mem copy time: " << elapsed << " seconds\n";


    for (int i = 0; i < 32; i++) {
        std::cout << pOutVector[i] << " ";
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }

    // Clean up
    delete[] pInVector;
    delete[] pOutVector;
    cudaFree(d_pInVector);
    cudaFree(d_pOutVector);
    cudaFree(d_vec_2);
    cudaFree(d_vec_1);

    return 0;
}
