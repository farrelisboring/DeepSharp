#pragma once
#include "exports.h"
#include "matrix.h"


extern "C" {
	MATRIX_API matrix_double* matrix_multiply_double(const matrix_double* __restrict__ a_stru, const matrix_double* __restrict__ b_stru, size_t a_row, size_t shared_dimension, size_t b_col);

	MATRIX_API matrix_float* matrix_multiply_float(const matrix_float* a_stru, const matrix_float* b_stru, size_t a_row, size_t shared_dimension, size_t b_col);
	MATRIX_API matrix_float* matrix_add_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc);
	MATRIX_API matrix_float* matrix_subtract_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc);
	MATRIX_API matrix_float* transpose_float(const matrix_float* __restrict__ a_struc);

	MATRIX_API matrix_half* matrix_multiply_half(const matrix_half* __restrict__ a_stru, const matrix_half* __restrict__ b_stru, size_t a_row, size_t shared_dimension, size_t b_col);
}