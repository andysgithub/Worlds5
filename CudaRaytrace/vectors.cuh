#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <numbers>

#define DimTotal 5

struct Vector3 {
    float X, Y, Z;

    __device__ __host__ Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : X(x), Y(y), Z(z) {}

    __device__ __host__ Vector3 operator+(const Vector3& v) const {
        return Vector3(X + v.X, Y + v.Y, Z + v.Z);
    }

    __device__ __host__ Vector3 operator*(float scalar) {
        return Vector3(scalar * X, scalar * Y, scalar * Z);
    }

    __device__ __host__ float dot(const Vector3& v) {
        return X * v.X + Y * v.Y + Z * v.Z;
    }

    __device__ __host__ Vector3 Normalize() const {
        const float magnitude = std::sqrt(X * X + Y * Y + Z * Z);
        if (magnitude > 0) {
            return Vector3(X / magnitude, Y / magnitude, Z / magnitude);
        }
        return *this;
    }
};


struct Vector5 {
    float m[5];

    // Constructor
    __device__ __host__ Vector5(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f, float v = 0.0f) {
        m[0] = x;
        m[1] = y;
        m[2] = z;
        m[3] = w;
        m[4] = v;
    }

    __device__ __host__ Vector5 operator+(const Vector5& v) const {
        return Vector5(m[0] + v.m[0], m[1] + v.m[1], m[2] + v.m[2], m[3] + v.m[3], m[4] + v.m[4]);
    }

    __device__ __host__ Vector5 operator-(const Vector5& v) const {
        return Vector5(m[0] - v.m[0], m[1] - v.m[1], m[2] - v.m[2], m[3] - v.m[3], m[4] - v.m[4]);
    }

    __device__ __host__ Vector5 operator*(float scalar) const {
        return Vector5(scalar * m[0], scalar * m[1], scalar * m[2], scalar * m[3], scalar * m[4]);
    }

    __device__ __host__ Vector5 operator*(const Vector5& o) const {
        return Vector5(
            m[0] * o.m[0] - m[1] * (o.m[1] - o.m[2] + o.m[3] - o.m[4]) + o.m[1] * (m[2] - m[3] + m[4]),
            m[0] * o.m[1] + m[1] * o.m[0] + m[2] * (o.m[2] - o.m[3] + o.m[4]) - o.m[2] * (m[3] - m[4]),
            m[0] * o.m[2] + m[2] * o.m[0] - m[3] * (o.m[3] - o.m[4]) + o.m[3] * m[4],
            m[0] * o.m[3] + m[3] * o.m[0] + m[4] * o.m[4],
            m[0] * o.m[4] + m[4] * o.m[0]
        );
    }

    __device__ __host__ Vector5 mandel(const Vector5& c) const {
        return Vector5(
            m[0] * m[0] - m[1] * m[1] + 2 * m[1] * (m[2] - m[3] + m[4]) + c.m[0],
            2 * (m[0] * m[1] - m[2] * m[3] + m[2] * m[4]) + m[2] * m[2] + c.m[1],
            2 * (m[0] * m[2] + m[3] * m[4]) - m[3] * m[3] + c.m[2],
            2 * m[0] * m[3] + m[4] * m[4] + c.m[3],
            2 * m[0] * m[4] + c.m[4]
        );
    }

    __device__ __host__ Vector5 normalize() const {
        const float mag = magnitude();
        if (mag > 0) {
            return Vector5(m[0] / mag, m[1] / mag, m[2] / mag, m[3] / mag, m[4] / mag);
        }
        return *this;
    }

    __device__ __host__ float dot(const Vector5& v) const {
        return m[0] * v.m[0] + m[1] * v.m[1] + m[2] * v.m[2] + m[3] * v.m[3] + m[4] * v.m[4];
    }

    __device__ __host__ float magnitude() const {
        return sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3] + m[4] * m[4]);
    }
};

__device__ __forceinline__ void mandel(float* __restrict__ a, const float* __restrict__ b) {
    float a0 = a[0];
    float a1 = a[1];
    float a2 = a[2];
    float a3 = a[3];
    float a4 = a[4];

    a[0] = a0 * a0 - a1 * a1 + 2 * a1 * (a2 - a3 + a4) + b[0];
    a[1] = 2 * (a0 * a1 - a2 * a3 + a2 * a4) + a2 * a2 + b[1];
    a[2] = 2 * (a0 * a2 + a3 * a4) - a3 * a3 + b[2];
    a[3] = 2 * a0 * a3 + a4 * a4 + b[3];
    a[4] = 2 * a0 * a4 + b[4];
}
