#include "matrix.h"
#include "utils.h"
#include "status.h"
#include <new>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

__global__ void kernel_fill_all_double(double* data, double x, size_t n) { //mybe fixes
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] = x;
}

extern "C" {

    matrix_double* matrix_double_create(size_t rows, size_t cols) {
        g_last_status = CudaResult_Success;
        const std::size_t n = safe_count(rows, cols);
        if (n == 0) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        matrix_double* m = new(std::nothrow) matrix_double;
        if (!m) {
            g_last_status = CudaResult_AllocationFailed;
            return nullptr;
        }
        m->rows = rows;
        m->cols = cols;

        double* data_addr;
        cudaError_t err = cudaMalloc((void**)&data_addr, n * sizeof(double));
        if (err != cudaSuccess) {
            delete m;
            g_last_status = CudaResult_CudaError;
            return nullptr;
        }
        m->data = data_addr;

        return m;
    }

    void matrix_free_double(matrix_double* m) {
        matrix_free_impl(m);
    }

    double matrix_getdouble(matrix_double* m, size_t row, size_t col) {
        g_last_status = CudaResult_Success;
        if (!m || !m->data || row >= m->rows || col >= m->cols) {
            g_last_status = CudaResult_InvalidArgument;
            return 0.0;
        }

        const std::size_t idx = (size_t)row * (size_t)m->cols + (size_t)col;
        double val{};
        cudaError_t err = cudaMemcpy(&val, m->data + idx, sizeof(double), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            return 0.0;
        }
        return val;
    }

    void fill_double_all_double(matrix_double* m, double x) {
        cudaStream_t stream = 0;
        g_last_status = CudaResult_Success;
        if (!m || !m->data) return;
        const std::size_t n = safe_count(m->rows, m->cols);

        init_launch_config(kernel_fill_all_double);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;


        kernel_fill_all_double <<<blocks, g_threads, 0, stream >> > (m->data, x, n);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
        }
    }




    matrix_float* matrix_float_create(size_t rows, size_t cols) {
        g_last_status = CudaResult_Success;
        const std::size_t n = safe_count(rows, cols);
        if (n == 0) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        matrix_float* m = new(std::nothrow) matrix_float;
        if (!m) {
            g_last_status = CudaResult_AllocationFailed;
            return nullptr;
        }
        m->rows = rows;
        m->cols = cols;

        float* data_addr;
        cudaError_t err = cudaMalloc((void**)&data_addr, n * sizeof(float));
        if (err != cudaSuccess) {
            delete m;
            g_last_status = CudaResult_CudaError;
            return nullptr;
        }
        m->data = data_addr;

        return m;
    }

    void matrix_free_float(matrix_float* m) {
        matrix_free_impl(m);
    }

    float matrix_getfloat(matrix_float* m, size_t rows, size_t cols) {
        g_last_status = CudaResult_Success;
        if (!m || !m->data || rows >= m->rows || cols >= m->cols) {
            g_last_status = CudaResult_InvalidArgument;
            return 0.0f;
        }

        const std::size_t n = (size_t)rows * (size_t)m->cols + (size_t)cols;
        float val{};
        cudaError_t err = cudaMemcpy(&val, m->data + n, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            return 0.0f;
        }
        return val;
    }

    
    matrix_half* matrix_half_create(size_t rows, size_t cols) {
        g_last_status = CudaResult_Success;
        const std::size_t n = safe_count(rows, cols);
        if (n == 0) {
            g_last_status = CudaResult_InvalidArgument;
            return nullptr;
        }

        matrix_half* m = new(std::nothrow) matrix_half;
        if (!m) {
            g_last_status = CudaResult_AllocationFailed;
            return nullptr;
        }
        m->rows = rows;
        m->cols = cols;

        half* data_addr;
        cudaError_t err = cudaMalloc((void**)&data_addr, n * sizeof(half));
        if (err != cudaSuccess) {
            delete m;
            g_last_status = CudaResult_CudaError;
            return nullptr;
        }
        m->data = data_addr;

        return m;
    }

    void matrix_free_half(matrix_half* m) {
        matrix_free_impl(m);
    }

    half matrix_gethalf(matrix_half* m, size_t rows, size_t cols) {
        g_last_status = CudaResult_Success;
        if (!m || !m->data || rows >= m->rows || cols >= m->cols) {
            g_last_status = CudaResult_InvalidArgument;
            return half{};
        }

        const std::size_t n = rows * m->cols + cols;
        half val{};
        cudaError_t err = cudaMemcpy(&val, m->data + n, sizeof(half), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            return half{};
        }
        return val;
    }
}