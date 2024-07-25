#pragma once

#ifndef VECTORS_H
#define VECTORS_H

#include <array>
#include <cmath>
#include <numbers>

#define DimTotal 5

struct Vector3 {
    float X, Y, Z;

    Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : X(x), Y(y), Z(z) {}

    Vector3 operator+(const Vector3& v) const {
        return Vector3(X + v.X, Y + v.Y, Z + v.Z);
    }

    Vector3 operator+(const float scalar) const {
        return Vector3(X + scalar, Y + scalar, Z + scalar);
    }

    Vector3 operator-(const Vector3& v) const {
        return Vector3(X - v.X, Y - v.Y, Z - v.Z);
    }

    Vector3 operator-(const float scalar) const {
        return Vector3(X - scalar, Y - scalar, Z - scalar);
    }

    Vector3 operator*(float scalar) {
        return Vector3(scalar * X, scalar * Y, scalar * Z);
    }

    float dot(const Vector3& v) {
        return X * v.X + Y * v.Y + Z * v.Z;
    }

    Vector3 normalize() const {
        const float mag = magnitude();
        if (mag > 0) {
            return Vector3(X / mag, Y / mag, Z / mag);
        }
        return *this;
    }

    float magnitude() const {
        return sqrtf(X * X + Y * Y + Z * Z);
    }
};


struct Vector5 {
    float m[5];

    // Constructor
    Vector5(float x = 0.0f, float y = 0.0f, float z = 0.0f, float w = 0.0f, float v = 0.0f) {
        m[0] = x;
        m[1] = y;
        m[2] = z;
        m[3] = w;
        m[4] = v;
    }

    Vector5 operator+(const Vector5& v) const {
        return Vector5(m[0] + v.m[0], m[1] + v.m[1], m[2] + v.m[2], m[3] + v.m[3], m[4] + v.m[4]);
    }

    Vector5 operator-(const Vector5& v) const {
        return Vector5(m[0] - v.m[0], m[1] - v.m[1], m[2] - v.m[2], m[3] - v.m[3], m[4] - v.m[4]);
    }

    Vector5 operator*(float scalar) const {
        return Vector5(scalar * m[0], scalar * m[1], scalar * m[2], scalar * m[3], scalar * m[4]);
    }

    Vector5 operator*(const Vector5& o) const {
        return Vector5(
            m[0] * o.m[0] - m[1] * (o.m[1] - o.m[2] + o.m[3] - o.m[4]) + o.m[1] * (m[2] - m[3] + m[4]),
            m[0] * o.m[1] + m[1] * o.m[0] + m[2] * (o.m[2] - o.m[3] + o.m[4]) - o.m[2] * (m[3] - m[4]),
            m[0] * o.m[2] + m[2] * o.m[0] - m[3] * (o.m[3] - o.m[4]) + o.m[3] * m[4],
            m[0] * o.m[3] + m[3] * o.m[0] + m[4] * o.m[4],
            m[0] * o.m[4] + m[4] * o.m[0]
        );
    }

    Vector5 mandel(const Vector5& c) const {
        return Vector5(
            m[0] * m[0] - m[1] * m[1] + 2 * m[1] * (m[2] - m[3] + m[4]) + c.m[0],
            2 * (m[0] * m[1] - m[2] * m[3] + m[2] * m[4]) + m[2] * m[2] + c.m[1],
            2 * (m[0] * m[2] + m[3] * m[4]) - m[3] * m[3] + c.m[2],
            2 * m[0] * m[3] + m[4] * m[4] + c.m[3],
            2 * m[0] * m[4] + c.m[4]
        );
    }

    Vector5 normalize() const {
        const float mag = magnitude();
        if (mag > 0) {
            return Vector5(m[0] / mag, m[1] / mag, m[2] / mag, m[3] / mag, m[4] / mag);
        }
        return *this;
    }

    float dot(const Vector5& v) const {
        return m[0] * v.m[0] + m[1] * v.m[1] + m[2] * v.m[2] + m[3] * v.m[3] + m[4] * v.m[4];
    }

    float magnitude() const {
        return sqrtf(m[0] * m[0] + m[1] * m[1] + m[2] * m[2] + m[3] * m[3] + m[4] * m[4]);
    }
};

void v_mandel(float* a, const float* b);

#endif
