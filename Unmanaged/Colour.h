#pragma once

#ifndef COLOUR_H
#define COLOUR_H

const float inv60 = (float)0.01667;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define COLOUR_API __declspec(dllexport)
#else
#define COLOUR_API
#endif

COLOUR_API void __stdcall HSVtoRGB(float h, float s, float v, unsigned char* rval, unsigned char* gval, unsigned char* bval);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
// C++ specific overload that uses references
inline void HSVtoRGB(float h, float s, float v, unsigned char& r, unsigned char& g, unsigned char& b) {
    HSVtoRGB(h, s, v, &r, &g, &b);
}
#endif

#endif // COLOUR_H