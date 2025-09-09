# Core Concepts

## Interop Layer Design

DeepSharp uses P/Invoke to call into a native library (`MatrixNative.dll`). Managed wrappers own safe handles that track GPU memory and translate calls into CUDA kernels.

## Memory Management

GPU allocations are represented by `SafeHandle` subclasses. Managed objects dispose these handles to release device memory deterministically.

## Error Handling

Every native call reports a status code inspected by `NativeMethods.ThrowIfFailed`. Failures surface as `CudaException` or `CurandException` to provide .NET-friendly diagnostics.
