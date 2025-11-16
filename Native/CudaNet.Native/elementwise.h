#pragma once
#include "exports.h"
#include "matrix.h"


extern "C" {

    MATRIX_API matrix_float* scalar_division_float(const matrix_float* __restrict__ a_struc, const float y);
    MATRIX_API matrix_float* scalar_multiplication_float(const matrix_float* __restrict__ a_struc, const float y);
    MATRIX_API matrix_float* broadcast_rowwise_subtract_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc);
    MATRIX_API matrix_float* broadcast_rowwise_add_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc);
    MATRIX_API matrix_float* scalar_subtract_float(const matrix_float* __restrict__ a_struc, const float y);
    MATRIX_API matrix_float* scalar_add_float(const matrix_float* __restrict__ a_struc, const float y);
    MATRIX_API matrix_float* elementwise_multiplication_float(const matrix_float* __restrict__ a_struc, const matrix_float* __restrict__ b_struc);
    MATRIX_API matrix_float* column_sum_float(const matrix_float* __restrict__  a_struc);

}