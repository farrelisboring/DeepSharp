# Usage Guides

## Tutorials

- Invoke basic matrix operations from C#.
- Chain activation functions like `TanhFloat` and `LeakyReluFloat`.

## Best Practices

- Dispose matrices to free GPU memory.
- Validate dimensions before launching kernels.
- Keep data on the GPU to avoid slow transfers.

## Debugging Tips

- Enable CUDA debug symbols when building native code.
- Use `ThrowIfFailed` messages to pinpoint failing operations.
- Check GPU memory usage with `nvidia-smi` during development.
