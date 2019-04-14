#ifndef _VECTORS_H
#define _VECTORS_H

typedef struct vectorDoubleType
{
    double coords[5];
}
vector5Double;

double vectorAngle(vector5Double A, vector5Double B, vector5Double C);

#endif