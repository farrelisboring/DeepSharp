#pragma once
#include "exports.h"
#include "matrix.h"


extern "C" {
	MATRIX_API void     fill_double_rng(matrix_double* m, unsigned long long seed);

	MATRIX_API void     fill_float_rng(matrix_float* m, unsigned long long seed);
	MATRIX_API void fill_kaiming_normal_float(matrix_float* m, int fan_in, unsigned long long seed, unsigned long long subseq /* optional */ = 0ULL);

	MATRIX_API void     fill_half_rng(matrix_half* m, unsigned long long seed, unsigned long long subseq);
}