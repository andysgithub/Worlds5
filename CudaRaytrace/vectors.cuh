#pragma once

#include <array>
#include <cmath>
#include <numbers>
#include <cuda_runtime.h>
//#include "Vector5.h"

//float vectorAngle(Vector5 A, Vector5 B, Vector5 C);

#define DimTotal 5

struct Vector3 {
    float X, Y, Z;

    __device__ Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : X(x), Y(y), Z(z) {}

    __device__ Vector3 operator+(const Vector3& v) const {
        return Vector3(X + v.X, Y + v.Y, Z + v.Z);
    }

    __device__ friend Vector3 operator*(float scalar, const Vector3& v) {
        return Vector3(scalar * v.X, scalar * v.Y, scalar * v.Z);
    }

    __device__ static float Dot(const Vector3& a, const Vector3& b) {
        return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
    }

    __device__ Vector3 Normalize() const {
        const float magnitude = std::sqrt(X * X + Y * Y + Z * Z);
        if (magnitude > 0) {
            return Vector3(X / magnitude, Y / magnitude, Z / magnitude);
        }
        return *this;
    }
};

#include <cmath>

struct Vector5 {
    float m[5];

    __host__ __device__ Vector5(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f, float v = 0.0f) {
        m[0] = x;
        m[1] = y;
        m[2] = z;
        m[3] = w;
        m[4] = v;
    }

    __host__ __device__ float& operator[](int index) {
        return m[index];
    }

    __host__ __device__ const float& operator[](int index) const {
        return m[index];
    }

    __host__ __device__ Vector5 operator+(const Vector5& o) const {
        Vector5 result;

        result.m[0] = m[0] + o.m[0];
        result.m[1] = m[1] + o.m[1];
        result.m[2] = m[2] + o.m[2];
        result.m[3] = m[3] + o.m[3];
        result.m[4] = m[4] + o.m[4];

        return result;
    }

    __host__ __device__ Vector5 operator-(const Vector5& o) const {
        Vector5 result;

        result.m[0] = m[0] - o.m[0];
        result.m[1] = m[1] - o.m[1];
        result.m[2] = m[2] - o.m[2];
        result.m[3] = m[3] - o.m[3];
        result.m[4] = m[4] - o.m[4];
        
        return result;
    }

    __host__ __device__ Vector5 operator*(float scalar) const {
        Vector5 result;

        result.m[0] = m[0] * scalar;
        result.m[1] = m[1] * scalar;
        result.m[2] = m[2] * scalar;
        result.m[3] = m[3] * scalar;
        result.m[4] = m[4] * scalar;

        return result;
    }

    __host__ __device__ Vector5 operator*(const Vector5& o) const {
        return Vector5(
            m[0] * o.m[0] - m[1] * (o.m[1] - o.m[2] + o.m[3] - o.m[4]) + o.m[1] * (m[2] - m[3] + m[4]),
            m[0] * o.m[1] + m[1] * o.m[0] + m[2] * (o.m[2] - o.m[3] + o.m[4]) - o.m[2] * (m[3] - m[4]),
            m[0] * o.m[2] + m[2] * o.m[0] - m[3] * (o.m[3] - o.m[4]) + o.m[3] * m[4],
            m[0] * o.m[3] + m[3] * o.m[0] + m[4] * o.m[4],
            m[0] * o.m[4] + m[4] * o.m[0]
        );
    }

    __host__ __device__ float dot(const Vector5& o) const {
        float result = 0;

        result += m[0] * o.m[0];
        result += m[1] * o.m[1];
        result += m[2] * o.m[2];
        result += m[3] * o.m[3];
        result += m[4] * o.m[4];

        return result;
    }

    __host__ __device__ Vector5 scale(const float s) const {
        return (*this) * s;
    }

    __host__ __device__ Vector5 normalize() const {
        const float mag = magnitude();
        if (mag > 0) {
            Vector5 result;

            result.m[0] = this->m[0] / mag;
            result.m[1] = this->m[1] / mag;
            result.m[2] = this->m[2] / mag;
            result.m[3] = this->m[3] / mag;
            result.m[4] = this->m[4] / mag;

            return result;
        }
        return *this;
    }

	__host__ __device__ float magnitude() const {
        return m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3] + m[4] * m[4];
    }

    __host__ __device__ static float clamp(float x, float lower, float upper) {
        return fminf(upper, fmaxf(x, lower));
    }
};

// Non-member operator for scalar * Vector5
__host__ __device__ inline Vector5 operator*(float scalar, const Vector5& v) {
    return v * scalar;
}
