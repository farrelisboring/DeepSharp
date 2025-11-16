#include "status.h"


CudaResult g_last_status = CudaResult_Success;

extern "C" CudaResult get_last_status() {
    return g_last_status;
}
