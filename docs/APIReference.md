# API Reference

## NativeMethods (P/Invoke)

- `matrix_double_create(int rows, int cols)` — allocate a double matrix on the GPU.
- `matrix_free(IntPtr m)` — free a double-precision matrix.
- `fill_double_rng(MatrixDoubleHandle m, ulong seed)` — fill with random doubles.
- `fill_double_all_double(MatrixDoubleHandle m, double value)` — fill with a constant.
- `matrix_getdouble(MatrixDoubleHandle m, int row, int col)` — read element.
- `matrix_multiply_double(MatrixDoubleHandle a, MatrixDoubleHandle b, int aRow, int sharedDim, int bCol)` — multiply two double matrices.
- `matrix_float_create(int rows, int cols)` — allocate a float matrix.
- `matrix_free_float(IntPtr m)` — free a float matrix.
- `matrix_multiply_float(MatrixFloatHandle a, MatrixFloatHandle b, int aRow, int sharedDim, int bCol)` — multiply float matrices.
- `fill_float_rng(MatrixFloatHandle m, ulong seed)` — fill with random floats.
- `matrix_getfloat(MatrixFloatHandle m, int row, int col)` — read float element.
- `tanh_float(MatrixFloatHandle m)` — apply tanh.
- `leaky_relu_float(MatrixFloatHandle m, float alpha)` — apply leaky ReLU.
- `d_leaky_relu_float(MatrixFloatHandle m, float alpha)` — derivative of leaky ReLU.
- `elementwise_multiplication_float(MatrixFloatHandle a, MatrixFloatHandle b)` — element-wise multiply.
- `transpose_float(MatrixFloatHandle a)` — transpose matrix.
- `sigmoid_float(MatrixFloatHandle m)` — apply sigmoid.
- `scalar_division_float(MatrixFloatHandle a, float y)` — divide by scalar.
- `scalar_multiplication_float(MatrixFloatHandle a, float y)` — multiply by scalar.
- `broadcast_rowwise_subtract_float(MatrixFloatHandle a, MatrixFloatHandle b)` — subtract row vector from each row.
- `broadcast_rowwise_add_float(MatrixFloatHandle a, MatrixFloatHandle b)` — add row vector to each row.
- `scalar_subtract_float(MatrixFloatHandle a, float y)` — subtract scalar.
- `scalar_add_float(MatrixFloatHandle a, float y)` — add scalar.
- `fill_kaiming_normal_float(MatrixFloatHandle m, int fanIn, ulong seed, ulong subseq)` — fill with Kaiming normal distribution.
- `matrix_add_float(MatrixFloatHandle a, MatrixFloatHandle b)` — add matrices.
- `matrix_subtract_float(MatrixFloatHandle a, MatrixFloatHandle b)` — subtract matrices.
- `column_sum_float(MatrixFloatHandle a)` — sum columns.
- `matrix_half_create(int rows, int cols)` — allocate half-precision matrix.
- `matrix_free_half(IntPtr m)` — free half matrix.
- `matrix_multiply_half(MatrixHalfHandle a, MatrixHalfHandle b, int aRow, int sharedDim, int bCol)` — multiply half matrices.
- `fill_half_rng(MatrixHalfHandle m, ulong seed, ulong subseq)` — fill half matrix with random values.
- `matrix_gethalf(MatrixHalfHandle m, int row, int col)` — read half-precision element.
- `get_last_status()` — return status code of last CUDA/curand call.

## NativeMatrixDouble

- `NativeMatrixDouble(int rows, int cols)` — construct matrix.
- `void FillRandom(ulong seed)` — fill with random doubles.
- `void Fill(double value)` — fill with constant.
- `double this[int r, int c]` — indexer.
- `static NativeMatrixDouble Multiply(NativeMatrixDouble a, NativeMatrixDouble b)` — multiply.
- `void Dispose()` — release resources.

## NativeMatrixFloat

- `NativeMatrixFloat(int rows, int cols)` — construct matrix.
- `float this[int r, int c]` — indexer.
- `void FillRandom(ulong seed)` — fill with random floats.
- `int GetRows()` / `int GetCols()` — query dimensions.
- `static NativeMatrixFloat SumColumn(NativeMatrixFloat a)` — column sums.
- `void FillKaimingNormal(int fanIn, ulong seed, ulong subseq=0)` — Kaiming init.
- `static NativeMatrixFloat LeakyReluFloat(NativeMatrixFloat a, float alpha)` — leaky ReLU.
- `static NativeMatrixFloat TanhFloat(NativeMatrixFloat a)` — tanh.
- `static NativeMatrixFloat DLeakyReluFloat(NativeMatrixFloat a, float alpha)` — derivative.
- `static NativeMatrixFloat ElementwiseMultiplyFloat(NativeMatrixFloat a, NativeMatrixFloat b)` — element-wise multiply.
- `static NativeMatrixFloat TransposeFloat(NativeMatrixFloat a)` — transpose.
- `static NativeMatrixFloat SigmoidFloat(NativeMatrixFloat a)` — sigmoid.
- `static NativeMatrixFloat ScalarDivisionFloat(NativeMatrixFloat a, float value)` — divide by scalar.
- `static NativeMatrixFloat ScalarMultiplicationFloat(NativeMatrixFloat a, float value)` — multiply by scalar.
- `static NativeMatrixFloat ScalarAddFloat(NativeMatrixFloat a, float value)` — add scalar.
- `static NativeMatrixFloat ScalarSubtractFloat(NativeMatrixFloat a, float value)` — subtract scalar.
- `static NativeMatrixFloat BroadcastRowwiseAddFloat(NativeMatrixFloat a, NativeMatrixFloat b)` — rowwise add.
- `static NativeMatrixFloat BroadcastRowwiseSubtractFloat(NativeMatrixFloat a, NativeMatrixFloat b)` — rowwise subtract.
- `static NativeMatrixFloat MatrixAddFloat(NativeMatrixFloat a, NativeMatrixFloat b)` — add matrices.
- `static NativeMatrixFloat MatrixSubtractFloat(NativeMatrixFloat a, NativeMatrixFloat b)` — subtract matrices.
- `static NativeMatrixFloat Multiply(NativeMatrixFloat a, NativeMatrixFloat b)` — multiply.
- `void Dispose()` — release resources.

## NativeMatrixHalf

- `NativeMatrixHalf(int rows, int cols)` — construct half matrix.
- `void FillRandom(ulong seed, ulong subseq)` — fill with random halves.
- `Half this[int r, int c]` — indexer.
- `static NativeMatrixHalf Multiply(NativeMatrixHalf a, NativeMatrixHalf b)` — multiply.
- `void Dispose()` — release resources.

## Error Types

- `CudaException` — thrown on CUDA runtime errors.
- `CurandException` — thrown on cuRAND errors.
