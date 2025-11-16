#include "linalg.h"
#include "utils.h"
#include "status.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include "matrix.h"



__global__ void kernel_matrix_add_float(const float* __restrict__ data, const float* __restrict__ data_second, float* __restrict__ data_destination, size_t n_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
#pragma unroll 4
    for (size_t i = idx; i < n_elements; i += stride) {
        data_destination[i] = data[i] + data_second[i];
    }
}

__global__ void kernel_matrix_subtract_float(const float* __restrict__ data, const float* __restrict__ data_second, float* __restrict__ data_destination, size_t n_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
#pragma unroll 4
    for (size_t i = idx; i < n_elements; i += stride) {
        data_destination[i] = data[i] - data_second[i];
    }
}


extern "C" {
    matrix_double* matrix_multiply_double(const matrix_double* __restrict__ a_stru, const matrix_double* __restrict__ b_stru, size_t a_row, size_t shared_dimension, size_t b_col) {
        g_last_status = CudaResult_Success;

        if (!a_stru || !b_stru || !a_stru->data || !b_stru->data) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        if (a_row != a_stru->rows || shared_dimension != a_stru->cols ||
            shared_dimension != b_stru->rows || b_col != b_stru->cols) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        matrix_double* c_stru = matrix_double_create(a_row, b_col);
        if (g_last_status != CudaResult_Success || !c_stru || !c_stru->data)
            return nullptr;

        cublasHandle_t handle{};
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            matrix_free_double(c_stru);
            return nullptr;
        }

        const double alpha = 1.0, beta = 0.0;

        stat = cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            /*m'*/ b_col, /*n'*/ a_row, /*k'*/ shared_dimension,
            &alpha,
            /*B*/ b_stru->data, /*lda'*/ b_col,
            /*A*/ a_stru->data, /*ldb'*/ shared_dimension,
            &beta,
            /*C*/ c_stru->data, /*ldc'*/ b_col);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            cublasDestroy(handle);
            matrix_free_double(c_stru);
            return nullptr;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            cublasDestroy(handle);
            matrix_free_double(c_stru);
            return nullptr;
        }

        cublasDestroy(handle);
        return c_stru;
    }

    matrix_float* matrix_multiply_float(const matrix_float* a_stru, const matrix_float* b_stru, size_t a_row, size_t shared_dimension, size_t b_col) {
        g_last_status = CudaResult_Success;

        if (!a_stru || !b_stru || !a_stru->data || !b_stru->data) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        if (a_row != a_stru->rows || shared_dimension != a_stru->cols ||
            shared_dimension != b_stru->rows || b_col != b_stru->cols) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        matrix_float* c_stru = matrix_float_create(a_row, b_col);
        if (g_last_status != CudaResult_Success || !c_stru || !c_stru->data)
            return nullptr;

        cublasHandle_t handle{};
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_stru);
            return nullptr;
        }

        const float alpha = 1.0f, beta = 0.0f;

        stat = cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            b_col, a_row, shared_dimension,
            &alpha,
            b_stru->data, b_col,
            a_stru->data, shared_dimension,
            &beta,
            c_stru->data, b_col);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            cublasDestroy(handle);
            matrix_free_float(c_stru);
            return nullptr;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            cublasDestroy(handle);
            matrix_free_float(c_stru);
            return nullptr;
        }

        cublasDestroy(handle);
        return c_stru;
    }

    matrix_float* matrix_add_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc) { // header
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (a_struc->data == nullptr) return nullptr;


        init_launch_config(kernel_matrix_add_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_matrix_add_float <<<blocks, g_threads, 0, stream >> > (a_struc->data, b_struc->data, c_struc->data, n);


        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* matrix_subtract_float(const matrix_float* __restrict__  a_struc, const matrix_float* __restrict__ b_struc) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (a_struc->data == nullptr) return nullptr;


        init_launch_config(kernel_matrix_subtract_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_matrix_subtract_float << <blocks, g_threads, 0, stream >> > (a_struc->data, b_struc->data, c_struc->data, n);


        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* transpose_float(const matrix_float* __restrict__ a_struc) {
        g_last_status = CudaResult_Success;

        matrix_float* c_struc = matrix_float_create(a_struc->cols, a_struc->rows);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr || c_struc->data == nullptr)
            return nullptr;

        const int m = static_cast<int>(a_struc->cols); 
        const int n = static_cast<int>(a_struc->rows); 
        const int lda = static_cast<int>(a_struc->rows); 
        const int ldb = static_cast<int>(a_struc->rows); 
        const int ldc = static_cast<int>(a_struc->cols); 
        const float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasStatus_t st = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, a_struc->data, lda, &beta, a_struc->data, ldb, c_struc->data, ldc);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            cublasDestroy(handle);
            return nullptr;
        }

        if (st != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            cublasDestroy(handle);
            return nullptr;
        }
        cublasDestroy(handle);
        return c_struc;
    }

    matrix_half* matrix_multiply_half(const matrix_half* __restrict__ a_stru, const matrix_half* __restrict__ b_stru,
        size_t a_row, size_t shared_dimension, size_t b_col) {
        g_last_status = CudaResult_Success;

        if (!a_stru || !b_stru || !a_stru->data || !b_stru->data) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        if (a_row != a_stru->rows || shared_dimension != a_stru->cols ||
            shared_dimension != b_stru->rows || b_col != b_stru->cols) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        matrix_half* c_stru = matrix_half_create(a_row, b_col);
        if (g_last_status != CudaResult_Success || !c_stru || !c_stru->data)
            return nullptr;

        cublasHandle_t handle{};
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            matrix_free_half(c_stru);
            return nullptr;
        }

        const half alpha = __float2half(1.0f), beta = __float2half(0.0f);

        stat = cublasHgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            b_col, a_row, shared_dimension,
            &alpha,
            b_stru->data, b_col,
            a_stru->data, shared_dimension,
            &beta,
            c_stru->data, b_col);

        if (stat != CUBLAS_STATUS_SUCCESS) {
            g_last_status = CudaResult_CudaError;
            cublasDestroy(handle);
            matrix_free_half(c_stru);
            return nullptr;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            cublasDestroy(handle);
            matrix_free_half(c_stru);
            return nullptr;
        }

        cublasDestroy(handle);
        return c_stru;
    }
}