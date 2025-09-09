#pragma once
#include "exports.h"

enum CudaResult {
    CudaResult_Success = 0,
    CudaResult_CudaError = 1,
    CudaResult_CurandError = 2,
    CudaResult_InvalidArgument = 3,
    CudaResult_AllocationFailed = 4
};

// Declaration only (shared across TUs)
extern CudaResult g_last_status;

// C linkage for public-facing API
extern "C" MATRIX_API CudaResult get_last_status();
