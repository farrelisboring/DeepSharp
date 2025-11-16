#include "random.h"
#include "status.h"
#include "utils.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <cmath>


__global__ void kernel_init_states(unsigned long seed, unsigned long subseq, curandStatePhilox4_32_10_t* states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, subseq, &states[idx]);
}

__global__ void kernel_fill_with_curand_half(__half* out, size_t n, curandStatePhilox4_32_10_t* states) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t local = states[idx];
    float val = curand_uniform(&local);
    out[idx] = __float2half_rn(val);
    states[idx] = local;
}


void fill_with_curand_host_half(half* d_out, size_t n, unsigned long long seed, unsigned long long subseq /* = 0ULL */)
{
    cudaStream_t stream = 0;
    g_last_status = CudaResult_Success;
    static curandStatePhilox4_32_10_t* d_states = nullptr;
    static size_t allocated = 0;            // number of states allocated

    int reqBlocks = (n + g_threads - 1) / g_threads;
    int blocks = (reqBlocks < g_maxBlocks) ? reqBlocks : g_maxBlocks;

    if (!d_states || allocated < n) {
        if (d_states) cudaFree(d_states);
        if (cudaMalloc(&d_states, n * sizeof(curandStatePhilox4_32_10_t)) != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            d_states = nullptr;
            allocated = 0;
            return;
        }
        kernel_init_states << <blocks, g_threads, 0, stream >> > (seed, subseq, d_states);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            return;
        }
        allocated = n;
    }

    kernel_fill_with_curand_half << <blocks, g_threads, 0, stream >> > (reinterpret_cast<__half*>(d_out), n, d_states);
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        g_last_status = CudaResult_CudaError;
    }
}

void fill_with_curand_host_float(float* d_out, size_t n,
    unsigned long long seed,
    unsigned long long subseq)
{
    g_last_status = CudaResult_Success;
    curandGenerator_t gen;
    curandStatus_t st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    if (st != CURAND_STATUS_SUCCESS) {
        g_last_status = CudaResult_CurandError;
        return;
    }

    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    if (st != CURAND_STATUS_SUCCESS) {
        g_last_status = CudaResult_CurandError;
        curandDestroyGenerator(gen);
        return;
    }

    if (subseq) curandSetGeneratorOffset(gen, (unsigned long long)n * subseq);

    st = curandGenerateUniform(gen, d_out, n);
    if (st != CURAND_STATUS_SUCCESS) {
        g_last_status = CudaResult_CurandError;
        curandDestroyGenerator(gen);
        return;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        g_last_status = CudaResult_CudaError;
        curandDestroyGenerator(gen);
        return;
    }

    curandDestroyGenerator(gen);
}

void fill_with_curand_host_double(double* d_out, size_t n, unsigned long long seed, unsigned long long subseq)
{
    g_last_status = CudaResult_Success;
    curandGenerator_t gen;
    curandStatus_t st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    if (st != CURAND_STATUS_SUCCESS) {
        g_last_status = CudaResult_CurandError;
        return;
    }

    st = curandSetPseudoRandomGeneratorSeed(gen, seed);
    if (st != CURAND_STATUS_SUCCESS) {
        g_last_status = CudaResult_CurandError;
        curandDestroyGenerator(gen);
        return;
    }

    if (subseq) curandSetGeneratorOffset(gen, (unsigned long long)n * subseq);

    st = curandGenerateUniformDouble(gen, d_out, n);
    if (st != CURAND_STATUS_SUCCESS) {
        g_last_status = CudaResult_CurandError;
        curandDestroyGenerator(gen);
        return;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        g_last_status = CudaResult_CudaError;
        curandDestroyGenerator(gen);
        return;
    }

    curandDestroyGenerator(gen);
}



extern "C" {

    void fill_double_rng(matrix_double* m, unsigned long long seed) {
        g_last_status = CudaResult_Success;
        const std::size_t n = safe_count(m->rows, m->cols);
        fill_with_curand_host_double(m->data, n, seed, /*subseq=*/0ULL);
    }

    void fill_float_rng(matrix_float* m, unsigned long long seed) {
        g_last_status = CudaResult_Success;
        const std::size_t n = safe_count(m->rows, m->cols);
        fill_with_curand_host_float(m->data, n, seed, /*subseq=*/0ULL);
    }

    void fill_kaiming_normal_float(matrix_float* m, size_t fan_in, unsigned long long seed, unsigned long long subseq)
    {

        const size_t n = safe_count(m->rows, m->cols);
        const float stddev = std::sqrt(2.0f / static_cast<float>(fan_in));

        g_last_status = CudaResult_Success;
        curandGenerator_t gen;
        curandStatus_t st = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
        if (st != CURAND_STATUS_SUCCESS) {
            g_last_status = CudaResult_CurandError;
            return;
        }

        st = curandSetPseudoRandomGeneratorSeed(gen, seed);
        if (st != CURAND_STATUS_SUCCESS) {
            g_last_status = CudaResult_CurandError;
            curandDestroyGenerator(gen);
            return;
        }

        if (subseq) curandSetGeneratorOffset(gen, (unsigned long long)n * subseq);

        st = curandGenerateNormal(gen, m->data, n, 0.0f, stddev);
        if (st != CURAND_STATUS_SUCCESS) {
            g_last_status = CudaResult_CurandError;
            curandDestroyGenerator(gen);
            return;
        }

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            g_last_status = CudaResult_CudaError;
            curandDestroyGenerator(gen);
            return;
        }

        curandDestroyGenerator(gen);
    }

    void fill_half_rng(matrix_half* m, unsigned long long seed, unsigned long long subseq) {
        g_last_status = CudaResult_Success;
        const std::size_t n = safe_count(m->rows, m->cols);
        fill_with_curand_host_half(m->data, n, seed, subseq);
    }
}





