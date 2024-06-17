#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

#define DimTotal 5

__constant__ double m_Trans[DimTotal][6];

struct vector5Double {
    double coords[DimTotal];
};

__device__ void VectorTrans(double x, double y, double z, vector5Double* c)
{
    for (int i = 0; i < DimTotal; i++)
    {
        (*c).coords[i] = m_Trans[i][0] * x +
            m_Trans[i][1] * y +
            m_Trans[i][2] * z +
            m_Trans[i][5];
    }
}



// Host function to copy the transformation matrix to device memory
void copyTransformationMatrixToDevice(double h_Trans[DimTotal][6]) {
    cudaMemcpyToSymbol(m_Trans, h_Trans, sizeof(double) * DimTotal * 6);
}

int main() {
    // Define the transformation matrix on the host
    double h_Trans[DimTotal][6] = {
        // Initialize with your transformation matrix values
    };

    // Copy the transformation matrix to the device
    copyTransformationMatrixToDevice(h_Trans);

    // Continue with the rest of your CUDA setup and kernel launches
    // ...

    return 0;
}
