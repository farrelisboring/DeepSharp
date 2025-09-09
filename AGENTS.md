# AGENTS

## Overview
DeepSharp is a hybrid project that links managed .NET code with native CUDA/C++ routines to accelerate matrix operations. The `Managed` directory contains the C# wrapper library, while `Native` holds the CUDA and C++ sources that provide the heavy-lifting math.

## Repository Layout
- `Managed/` – .NET class library and managed wrappers.
- `Native/` – Native CUDA/C++ implementation.

## Development Guidelines
- Follow standard C# conventions: PascalCase for types and methods, camelCase for locals and parameters.
- Document interop boundaries and memory ownership semantics.
- In native code, prefer RAII and avoid unchecked pointer arithmetic.
- Keep commits small and focused.

## Testing
- Build the managed project: `dotnet build Managed/NetTorch.sln`.
- Add tests or samples when introducing new features. (optional)

## Pull Request Requirements
- Clearly describe motivation, changes, and testing in the PR body.
- Reference related issues when applicable.
- Ensure the working tree is clean and all builds/tests pass before submitting, except for the native code because OPENAI's Codex does not support nvcc or gcc.

## Safety Considerations & Best Practices
- Validate all external inputs before passing them to native code to prevent buffer overflows or undefined behavior.
- Dispose of `SafeHandle` and other unmanaged resources deterministically.
- Avoid exposing raw pointers or memory addresses in public APIs.
- Review for thread-safety when adding parallelism or shared state.

