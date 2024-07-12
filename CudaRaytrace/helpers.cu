#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include "vector5Single.h"
#include "Vectors.h"
#include "RayTracer.cuh"

// Helper function to check CUDA errors
void checkCudaError2(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

// Vector5 ImageToFractalSpace2(float distance, Vector3 coord)
//{
//    // Determine the x,y,z coord for this point
//    const float XPos = distance * coord.X;
//    const float YPos = distance * coord.Y;
//    const float ZPos = distance * coord.Z;
//
//    vector5Single c = { 0,0,0,0,0 };
//
//    // Transform 3D point x,y,z into nD fractal space at point c[]
//    RayTracer::VectorTrans2(XPos, YPos, ZPos, &c);
//
//    // Return the 5D fractal space point
//    std::array<float, 5> arr = c.toArray();
//    return Vector5(arr[0], arr[1], arr[2], arr[3], arr[4]);
//}