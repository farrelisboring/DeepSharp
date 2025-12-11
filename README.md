# DeepSharp

DeepSharp is a machine learning interop library bridging CUDA/C++ performance with C# simplicity.

```mermaid
flowchart LR
 subgraph App["C# Application"]
        userCode["User Code"]
  end
 subgraph Managed["Managed Enviroment"]
        matrixAPI["NetTorch API"]
        csharpLayer["Managed Interop"]
  end
 subgraph Native["Native Enviroment"]
        interop["C++ Interop Layer"]
        kernels["CUDA Kernels"]
  end
 subgraph GPU["NVIDIA GPU"]
        deviceMem["Hardware Enviroment"]
  end
    userCode --> matrixAPI
    matrixAPI --> csharpLayer
    csharpLayer--> interop
    interop --> kernels
    kernels --> deviceMem
    deviceMem  --> kernels
    interop .-> csharpLayer
    matrixAPI .-> userCode

```

## Quickstart

```csharp
using DeepSharp;

var a = new NativeMatrixFloat(2, 2);
var b = new NativeMatrixFloat(2, 2);
a.FillRandom(42);
b.FillRandom(42);
var c = NativeMatrixFloat.Multiply(a, b);
Console.WriteLine(c[0, 0]);
```

## Installation & Build

*TODO: Add instructions for building native and managed components.*

## Documentation

- [Introduction](docs/Introduction.md)
- [API Reference](docs/APIReference.md)
- [Getting Started](docs/GettingStarted.md)
- [Core Concepts](docs/CoreConcepts.md)
