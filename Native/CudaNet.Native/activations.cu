#include "linalg.h"
#include "utils.h"
#include "status.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void kernel_sigmoid_float(const float* __restrict__ logits, const size_t n, float* data_dest) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (; i < n; i += stride) {
        float x = logits[i];
        // Branchless, fast math
        data_dest[i] = __fdividef(1.0f, 1.0f + __expf(-x));
    }
}

__global__ void kernel_tanh_float(const float* __restrict__ data, size_t n, float* data_dest) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        float x = data[i];
        data_dest[i] = tanhf(x);
    }
}

__global__ void kernel_leaky_relu_float(const float* __restrict__ data,
    size_t n, const float alpha, float* data_dest) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;


    for (size_t i = idx; i < n; i += stride) {
        float x = data[i];

        data_dest[i] = (x >= 0.0f) ? x : (alpha * x);
    }
}

__global__ void kernel_d_leaky_relu_float(const float* __restrict__ data,
    size_t n,
    const float alpha, float* data_dest) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x;


    for (size_t i = idx; i < n; i += stride) {
        float x = data[i];

        data_dest[i] = (x > 0.0f) ? 1.0f : alpha;
    }
}



extern "C" {
    matrix_float* sigmoid_float(const matrix_float* __restrict__ a_struc) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr || c_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_sigmoid_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_sigmoid_float << <blocks, g_threads, 0, stream >> > (a_struc->data, n, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* tanh_float(const matrix_float* __restrict__ a_struc) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr || c_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_tanh_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_tanh_float << <blocks, g_threads, 0, stream >> > (a_struc->data, n, c_struc->data);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* d_leaky_relu_float(const matrix_float* __restrict__ a_struc, const float alpha) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);
        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr || c_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_d_leaky_relu_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_d_leaky_relu_float << <blocks, g_threads, 0, stream >> > (a_struc->data, n, alpha, c_struc->data);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }

    matrix_float* leaky_relu_float(const matrix_float* __restrict__ a_struc, const float alpha) {
        g_last_status = CudaResult_Success;
        cudaStream_t stream = 0;

        const size_t n = safe_count(a_struc->cols, a_struc->rows);

        matrix_float* c_struc = matrix_float_create(a_struc->rows, a_struc->cols);
        if (g_last_status != CudaResult_Success || a_struc->data == nullptr || c_struc->data == nullptr)
            return nullptr;

        init_launch_config(kernel_leaky_relu_float);

        int reqBlocks = (n + g_threads - 1) / g_threads;
        int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

        kernel_leaky_relu_float << <blocks, g_threads, 0, stream >> > (a_struc->data, n, alpha, c_struc->data);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            matrix_free_float(c_struc);
            return nullptr;
        }
        return c_struc;
    }
}