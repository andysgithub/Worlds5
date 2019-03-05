#ifndef _VECTORS_H
#define _VECTORS_H

typedef struct vectorDoubleType
{
    double coords[5];
}
vector5Double;

//typedef struct vectorSingleType
//{
//    float coords[5];
//}
//vector5Single;

//// Subtract vector A from vector B
//vector5Double vectorSub(vector5Double A, vector5Double B);

//// Return the dot product of v1 and v2
//double dot(vector5Double v1, vector5Double v2);
//
//// Return the modulus of vector v
//double mod(vector5Double v);

// Return the normalised vector of v
//vector5Double norm(vector5Double v);

double vectorAngle(vector5Double A, vector5Double B, vector5Double C);

#endif