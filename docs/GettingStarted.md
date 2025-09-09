# Getting Started

## Prerequisites

- .NET 6 SDK or newer
- CUDA Toolkit 11.x+
- A C++ compiler supported by the CUDA toolkit
- A compatible NVIDIA GPU

## First Example

```csharp
using DeepSharp;

var a = new NativeMatrixFloat(2, 2);
var b = new NativeMatrixFloat(2, 2);
a.FillRandom(123);
b.FillRandom(456);
var c = NativeMatrixFloat.Multiply(a, b);
Console.WriteLine(c[0, 0]);
```

This snippet allocates matrices on the GPU and multiplies them via a CUDA kernel exposed through the C# API.
