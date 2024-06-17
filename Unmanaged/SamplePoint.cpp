__device__ void VectorTrans(double XPos, double YPos, double ZPos, vector5Double* c);
__device__ bool ProcessPoint(float* Modulus, float* Angle, vector5Double c);

__device__ bool SamplePoint(double distance, float* Modulus, float* Angle, double xFactor, double yFactor, double zFactor, vector5Double c)
{
    // Determine the x,y,z coordinates for this point
    const double XPos = distance * xFactor;
    const double YPos = distance * yFactor;
    const double ZPos = distance * zFactor;

    // Transform 3D point x,y,z into nD fractal space at point c
    VectorTrans(XPos, YPos, ZPos, &c);

    // Determine orbit value for this point
    return ProcessPoint(Modulus, Angle, c) ? 1 : 0;
}