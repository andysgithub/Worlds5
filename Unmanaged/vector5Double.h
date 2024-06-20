#ifndef VECTOR5DOUBLE_H
#define VECTOR5DOUBLE_H

#include <array>

// Declaration of DimTotal
extern int DimTotal;

struct vector5Double {
    double coords[5];

    // Method to convert the vector to a 5D array
    std::array<double, 5> toArray() const {
        return { coords[0], coords[1], coords[2], coords[3], coords[4] };
    }
};

#endif // VECTOR5DOUBLE_H