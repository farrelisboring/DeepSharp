#pragma once

#include "exports.h"
#include <cuda_fp16.h> 


extern "C" {



    // simple POD to hold metadata + pointer
    struct matrix_double {
        int rows;
        int cols;
        double* data;
    };

    struct matrix_float {
        int rows;
        int cols;
        float* data;
    };

    struct matrix_half {
        int rows;
        int cols;
        half* data;
    };

    MATRIX_API matrix_double* matrix_double_create(size_t rows, size_t cols);
    MATRIX_API void     matrix_free_double(matrix_double* m);
    MATRIX_API double   matrix_getdouble(matrix_double* m, size_t row, size_t col);
    MATRIX_API void     fill_double_all_double(matrix_double* m, double x);

    MATRIX_API matrix_float* matrix_float_create(size_t rows, size_t cols);
    MATRIX_API void     matrix_free_float(matrix_float* m);
    MATRIX_API float   matrix_getfloat(matrix_float* m, size_t rows, size_t cols);

    MATRIX_API matrix_half* matrix_half_create(size_t rows, size_t cols);
    MATRIX_API void     matrix_free_half(matrix_half* m);
    MATRIX_API half   matrix_gethalf(matrix_half* m, size_t rows, size_t cols);
}