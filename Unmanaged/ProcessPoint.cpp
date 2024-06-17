#include <cmath>

// Define the vector5Double structure
struct vector5Double {
    double coords[5];
};

__device__ void v_mov(const double* src, double* dest);
__device__ void v_mandel(double* z, const double* c);
__device__ void v_subm(const double* a, const double* b, double* result);
__device__ double v_mod(const double* a);
__device__ double vectorAngle(const vector5Double& a, const vector5Double& b, const vector5Double& c);

__device__ bool ProcessPoint(float* Modulus, float* Angle, vector5Double c)
{
    double const PI = 3.1415926536;
    double const PI_OVER_2 = PI / 2;

    const long MaxCount = 1000;  // Iteration count for external points
    vector5Double z;             // Temporary 5-D vector
    vector5Double diff;          // Temporary 5-D vector for orbit size
    double ModulusTotal = 0;
    double ModVal = 0;
    double AngleTotal = PI;      // Angle for first two vectors is 180 degrees
    long count;

    z.coords[3] = 0;
    z.coords[4] = 0;

    v_mov(c.coords, z.coords);            // z = c
    vector5Double vectorSet[3];           // Collection of the three most recent vectors for determining the angle between them
    v_mov(z.coords, vectorSet[1].coords); // Store the first point in the vector set

    for (count = 0; count < MaxCount; count++)
    {
        v_mandel(z.coords, c.coords);              // z = z*z + c
        v_mov(z.coords, vectorSet[2].coords);      // Store the new point in the vector set

        // Determine vector angle for the last three positions
        if (count > 0 && count < 10)
        {
            // Add angle to current total
            AngleTotal += vectorAngle(vectorSet[0], vectorSet[1], vectorSet[2]);
        }

        // Determine modulus for this point in orbit
        v_subm(c.coords, z.coords, diff.coords);   // Current orbit size = mod(z - c)
        ModVal = v_mod(diff.coords);

        // Accumulate modulus value
        ModulusTotal += ModVal;

        // Stop accumulating values when modulus exceeds bailout value
        if (ModVal > m_Bailout)
        {
            count++;
            break;
        }

        // Move the vectors down in the list for the next angle
        v_mov(vectorSet[1].coords, vectorSet[0].coords);
        v_mov(vectorSet[2].coords, vectorSet[1].coords);
    }

    // Calculate the average modulus over the orbit
    *Modulus = (float)(ModulusTotal / count);
    // Calculate the average angle over the orbit
    *Angle = (float)(AngleTotal / (count > 10 ? 10 : count + 1));
    // Return true if this point is external to the set
    return (count < MaxCount);
}