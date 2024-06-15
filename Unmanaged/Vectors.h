#ifndef _VECTORS_H
#define _VECTORS_H

#include <array>

typedef struct vectorDoubleType
{
    double coords[5];

    // Method to convert the vector to a 5D array
    std::array<double, 5> toArray() const {
        return { coords[0], coords[1], coords[2], coords[3], coords[4] };
    }
}
vector5Double;

double vectorAngle(vector5Double A, vector5Double B, vector5Double C);

#endif