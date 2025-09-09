#pragma once


#ifdef _WIN32
#ifdef MATRIXNATIVE_EXPORTS
#define MATRIX_API __declspec(dllexport)
#else
#define MATRIX_API __declspec(dllimport)
#endif
#else
#define MATRIX_API
#endif