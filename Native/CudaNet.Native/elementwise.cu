#include "elementwise.h"
#include "utils.h"
#include "status.h"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void kernel_scalar_division(const float* __restrict__ data_inc, const size_t n, const float y, float* data_dest) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; i < n; i += stride) {
        float x = data_inc[i];
        data_dest[i] = __fdividef(x, y);
    }
}

__global__ void kernel_scalar_multiplication(const float* __restrict__ data_inc, const size_t n, const float y, float* data_dest) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; i < n; i += stride) {
        float x = data_inc[i];
        data_dest[i] = __fmul_rn(x, y);
    }
}

__global__ void kernel_scalar_add(const float* __restrict__ data_inc, const size_t n, const float y, float* data_dest) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; i < n; i += stride) {
        float x = data_inc[i];
        data_dest[i] = x + y;
    }
}

__global__ void kernel_scalar_subtract(const float* __restrict__ data_inc, const size_t n, const float y, float* data_dest) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; i < n; i += stride) {
        float x = data_inc[i];
        data_dest[i] = x - y;
    }
}

__global__ void kernel_add_broadcast_rowwise(const float* __restrict__ a, const size_t a_rows, const size_t a_cols, const float* __restrict__ b, float* __restrict__ out)
{
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a_cols) return;


    const size_t start_row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t row_stride = blockDim.y * gridDim.y;

    size_t idx = start_row * a_cols + col;
    const size_t di = row_stride * a_cols;

    for (size_t row = start_row; row < a_rows; row += row_stride, idx += di) {
        out[idx] = a[idx] + b[col]; 
    }
}

__global__ void kernel_subtract_broadcast_rowwise(const float* __restrict__ a, const size_t a_rows, const size_t a_cols, const float* __restrict__ b, float* __restrict__ out)
{
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a_cols) return;



    const size_t start_row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t row_stride = blockDim.y * gridDim.y;


    size_t idx = start_row * a_cols + col;
    const size_t di = row_stride * a_cols;

    for (size_t row = start_row; row < a_rows; row += row_stride, idx += di) {
        out[idx] = a[idx] - b[col]; 
    }
}

__global__ void kernel_column_sum_float(const float* __restrict__ data, size_t rows, size_t cols, float* __restrict__ data_destination)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t col = tid; col < cols; col += stride) {
        float acc = 0.0f;
        size_t idx = col;

        #pragma unroll 4
        for (size_t r = 0; r < rows; ++r) {

            acc += data[idx];
            idx += cols;
        }
        data_destination[col] = acc;
    }
}

__global__ void kernel_elementwise_multiply_float (
    const float* __restrict__ data_a,
    const float* __restrict__ data_b,
    float* __restrict__ data_dest,
    const size_t n)
{

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vec_n = n / 4;


    const float4* vec_a = reinterpret_cast<const float4*>(data_a);
    const float4* vec_b = reinterpret_cast<const float4*>(data_b);
    float4* vec_dest = reinterpret_cast<float4*>(data_dest);


    for (size_t i = tid; i < vec_n; i += blockDim.x * gridDim.x) {
        float4 a = vec_a[i];
        float4 b = vec_b[i];
        float4 result;


        result.x = a.x * b.x;
        result.y = a.y * b.y;
        result.z = a.z * b.z;
        result.w = a.w * b.w;

        vec_dest[i] = result;
    }


    const size_t remainder_start = vec_n * 4;
    const size_t remainder_tid = tid + remainder_start;
    if (remainder_tid < n) {
        data_dest[remainder_tid] = data_a[remainder_tid] * data_b[remainder_tid];
    }
}


extern "C" {
    matrix_float* scalar_division_float(const matrix_float* a_struc, const float y) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_scalar_division);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_scalar_division << <blocks, g_threads, 0, stream >> > (a_struc->data, n, y, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* scalar_multiplication_float(const matrix_float* a_struc, const float y) { // header
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr)
            return nullptr;


        init_launch_config(kernel_scalar_multiplication);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_scalar_multiplication << <blocks, g_threads, 0, stream >> > (a_struc->data, n, y, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* scalar_add_float(const matrix_float* a_struc, const float y) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_scalar_add);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_scalar_add << <blocks, g_threads, 0, stream >> > (a_struc->data, n, y, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* scalar_subtract_float(const matrix_float* a_struc, const float y) { // header
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_scalar_subtract);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_scalar_subtract << <blocks, g_threads, 0, stream >> > (a_struc->data, n, y, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* broadcast_rowwise_add_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc) { // header
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (a_struc->data == nullptr) return nullptr;

        init_launch_config(kernel_add_broadcast_rowwise);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_add_broadcast_rowwise << <blocks, g_threads, 0, stream >> > (a_struc->data, a_struc->rows, a_struc->cols, b_struc->data, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* broadcast_rowwise_subtract_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc) { // header
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (a_struc->data == nullptr) return nullptr;


        init_launch_config(kernel_subtract_broadcast_rowwise);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_subtract_broadcast_rowwise << <blocks, g_threads, 0, stream >> > (a_struc->data, a_struc->rows, a_struc->cols, b_struc->data, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* column_sum_float(const matrix_float* __restrict__  a_struc) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (a_struc->data == nullptr) return nullptr;


        init_launch_config(kernel_column_sum_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_column_sum_float << <blocks, g_threads, 0, stream >> > (a_struc->data, a_struc->rows, a_struc->cols, c_struc->data);


        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* elementwise_multiplication_float(const matrix_float* a_struc, const matrix_float* b_struc) { // header
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);
        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);

        if (g_last_status != CudaResult_Success || a_struc->data == nullptr || b_struc->data == nullptr || c_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_elementwise_multiply_float);

        const int threads = g_threads;
        const size_t vec_n = (n + 3) / 4; 

        int reqBlocks = static_cast<int>((vec_n + threads - 1) / threads);
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;
        if (blocks <= 0) blocks = 1;

        kernel_elementwise_multiply_float << <blocks, threads, 0, stream >> > (a_struc->data, b_struc->data, c_struc->data, n);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }
}