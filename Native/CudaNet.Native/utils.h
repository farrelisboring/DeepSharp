#pragma once
#include <cstddef>
#include <cstdint>
#include "status.h"

inline int g_threads = 0;
inline int g_blocksPerSM = 0;
inline int g_maxBlocks = 0;

static inline std::size_t safe_count(std::size_t r, std::size_t c) {
    if (r == 0 || c == 0) return 0;
    if (r > SIZE_MAX / c) return 0;
    return r * c;
}

template <typename MatrixT>
static inline void matrix_free_impl(MatrixT* m) {
    cudaError_t err = cudaSuccess;
    if (m) {
        err = cudaFree(m->data);
        delete m;
    }
    g_last_status = (err == cudaSuccess) ? CudaResult_Success : CudaResult_CudaError;
}

inline void init_launch_config(const void* kernel_func) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);

    g_threads = (prop.maxThreadsPerBlock >= 512) ? 512 : prop.maxThreadsPerBlock;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&g_blocksPerSM, kernel_func, g_threads, 0);

    g_maxBlocks = g_blocksPerSM * prop.multiProcessorCount;
}
