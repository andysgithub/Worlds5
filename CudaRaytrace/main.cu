#include <iostream>
#include "kernel.cuh"

// Host function to copy the transformation matrix to device memory
void copyTransformationMatrixToDevice(double h_Trans[DimTotal][6]);

int main() {
    // Define the transformation matrix on the host
    double h_Trans[DimTotal][6] = {
        {1, 0, 0, 0, 0, 1},
        {0, 1, 0, 0, 0, 1},
        {0, 0, 1, 0, 0, 1},
        {0, 0, 0, 1, 0, 1},
        {0, 0, 0, 0, 1, 1}
    };

    // Copy the transformation matrix to the device
    copyTransformationMatrixToDevice(h_Trans);

    // Allocate memory for results on the device
    int rayPoints = 1000; // Example value
    int maxSamples = 10000; // Example value
    int* d_externalPoints;
    float* d_modulusValues;
    float* d_angles;
    double* d_distances;

    cudaMalloc(&d_externalPoints, rayPoints * sizeof(int));
    cudaMalloc(&d_modulusValues, rayPoints * sizeof(float));
    cudaMalloc(&d_angles, rayPoints * sizeof(float));
    cudaMalloc(&d_distances, rayPoints * sizeof(double));

    // Launch the kernel
    int blockSize = 256;
    int numBlocks = (rayPoints + blockSize - 1) / blockSize;
    TraceRayKernel<<<numBlocks, blockSize>>>(1.0, 0.1, 1.0, 0.1,
        1.0, 0.0, 0.0, 8.0,
        d_externalPoints, d_modulusValues, d_angles, d_distances,
        rayPoints, maxSamples, 0.1, 10, 0);

    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_externalPoints);
    cudaFree(d_modulusValues);
    cudaFree(d_angles);
    cudaFree(d_distances);

    return 0;
}