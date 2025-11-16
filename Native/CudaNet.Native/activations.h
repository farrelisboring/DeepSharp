#pragma once
#include "exports.h"
#include "matrix.h"


extern "C" {
	MATRIX_API matrix_float* sigmoid_float(const matrix_float* __restrict__ a_struc);
	MATRIX_API matrix_float* tanh_float(const matrix_float* __restrict__ a_struc);
	MATRIX_API matrix_float* leaky_relu_float(const matrix_float* __restrict__ a_struc, const float alpha);
	MATRIX_API matrix_float* d_leaky_relu_float(const matrix_float* __restrict__ a_struc, const float alpha);
}