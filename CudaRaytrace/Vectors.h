#pragma once

#define _USE_MATH_DEFINES

#include <array>
#include <cmath>
#include "vector5Single.h"

float vectorAngle(vector5Single A, vector5Single B, vector5Single C);

struct Vector3 {
    float X, Y, Z;

    Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : X(x), Y(y), Z(z) {}

    Vector3 operator+(const Vector3& v) const {
        return Vector3(X + v.X, Y + v.Y, Z + v.Z);
    }

    friend Vector3 operator*(float scalar, const Vector3& v) {
        return Vector3(scalar * v.X, scalar * v.Y, scalar * v.Z);
    }

    static float Dot(const Vector3& a, const Vector3& b) {
        return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    }

    Vector3 Normalize() const {
        const float magnitude = std::sqrt(X * X + Y * Y + Z * Z);
        if (magnitude > 0) {
            return Vector3(X / magnitude, Y / magnitude, Z / magnitude);
        }
        return *this;
    }
};

struct Vector5 {
    float X, Y, Z, W, V;

    Vector5(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f, float v = 0.0f)
        : X(x), Y(y), Z(z), W(w), V(v) {}

    Vector5 operator+(const Vector5& v) const {
        return Vector5(X + v.X, Y + v.Y, Z + v.Z, W + v.W, V + v.V);
    }

    Vector5 operator-(const Vector5& v) const {
        return Vector5(X - v.X, Y - v.Y, Z - v.Z, W - v.W, V - v.V);
    }

    friend Vector5 operator*(float scalar, const Vector5& v) {
        return Vector5(scalar * v.X, scalar * v.Y, scalar * v.Z, scalar * v.W, scalar * v.V);
    }

    static float Dot(const Vector5& v1, const Vector5& v2) {
        return v1.X * v2.X + v1.Y * v2.Y + v1.Z * v2.Z + v1.W * v2.W + v1.V * v2.V;
    }

    Vector5 Normalize() const {
        const float magnitude = Magnitude();
        if (magnitude > 0) {
            return Vector5(X / magnitude, Y / magnitude, Z / magnitude, W / magnitude, V / magnitude);
        }
        return *this;
    }

    float Magnitude() const {
        return std::sqrt(X * X + Y * Y + Z * Z + W * W + V * V);
    }
};
