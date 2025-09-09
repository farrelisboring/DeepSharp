namespace NetTorch;

using System;
using System.Runtime.InteropServices;

internal enum CudaResult
{
    Success = 0,
    CudaError = 1,
    CurandError = 2,
    InvalidArgument = 3,
    AllocationFailed = 4
}

internal static class NativeMethods
{
    private const string LibraryName = "MatrixNative";

    // Double precision API
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixDoubleHandle matrix_double_create(int rows, int cols);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void matrix_free(IntPtr m);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void fill_double_rng(MatrixDoubleHandle m, ulong seed);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void fill_double_all_double(MatrixDoubleHandle m, double value);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern double matrix_getdouble(MatrixDoubleHandle m, int row, int col);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixDoubleHandle matrix_multiply_double(
        MatrixDoubleHandle a,
        MatrixDoubleHandle b,
        int aRow,
        int sharedDim,
        int bCol);

    // Single precision API
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle matrix_float_create(int rows, int cols);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void matrix_free_float(IntPtr m);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle matrix_multiply_float(
        MatrixFloatHandle a,
        MatrixFloatHandle b,
        int aRow,
        int sharedDim,
        int bCol);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void fill_float_rng(MatrixFloatHandle m, ulong seed);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern float matrix_getfloat(MatrixFloatHandle m, int row, int col);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle tanh_float(MatrixFloatHandle m);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle leaky_relu_float(MatrixFloatHandle m, float alpha);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle d_leaky_relu_float(MatrixFloatHandle m, float alpha);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle elementwise_multiplication_float(
        MatrixFloatHandle a,
        MatrixFloatHandle b);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle transpose_float(MatrixFloatHandle a);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle sigmoid_float(MatrixFloatHandle m);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle scalar_division_float(MatrixFloatHandle a, float y);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle scalar_multiplication_float(MatrixFloatHandle a, float y);



    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle broadcast_rowwise_subtract_float(
        MatrixFloatHandle a,
        MatrixFloatHandle b);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle broadcast_rowwise_add_float(
        MatrixFloatHandle a,
        MatrixFloatHandle b);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle scalar_subtract_float(MatrixFloatHandle a, float y);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle scalar_add_float(MatrixFloatHandle a, float y);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void fill_kaiming_normal_float(
        MatrixFloatHandle m,
        int fanIn,
        ulong seed,
        ulong subseq);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle matrix_add_float(
        MatrixFloatHandle a,
        MatrixFloatHandle b);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle matrix_subtract_float(
        MatrixFloatHandle a,
        MatrixFloatHandle b);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixFloatHandle column_sum_float(MatrixFloatHandle a);



    // HALF
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixHalfHandle matrix_half_create(int rows, int cols); // TODO: Needs a half class

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void matrix_free_half(IntPtr m);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern MatrixHalfHandle matrix_multiply_half(
        MatrixHalfHandle a,
        MatrixHalfHandle b,
        int aRow,
        int sharedDim,
        int bCol);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void fill_half_rng(MatrixHalfHandle m, ulong seed, ulong subseq);

    // Returns raw 16-bit representation to avoid marshalling issues with System.Half
    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern ushort matrix_gethalf(MatrixHalfHandle m, int row, int col);

    [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
    internal static extern CudaResult get_last_status();

    internal static void ThrowIfFailed(string operation)
    {
        CudaResult status = get_last_status();
        if (status == CudaResult.Success) return;
        switch (status)
        {
            case CudaResult.CudaError:
                throw new CudaException($"{operation} failed due to CUDA error.");
            case CudaResult.CurandError:
                throw new CurandException($"{operation} failed due to cuRAND error.");
            case CudaResult.InvalidArgument:
                throw new ArgumentException($"{operation} received invalid argument.");
            case CudaResult.AllocationFailed:
                throw new OutOfMemoryException($"{operation} failed to allocate GPU memory.");
            default:
                throw new InvalidOperationException($"{operation} failed with unknown error.");
        }
    }
}

public class CudaException : Exception
{
    public CudaException(string message) : base(message) { }
}

public class CurandException : Exception
{
    public CurandException(string message) : base(message) { }
}

/// <summary>
/// Safe handle for a native matrix_double*.
/// The underlying memory resides on the GPU and must be released via matrix_free.
/// </summary>
public sealed class MatrixDoubleHandle : SafeHandle
{
    private MatrixDoubleHandle() : base(IntPtr.Zero, ownsHandle: true) { }
    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.matrix_free(handle);
        return true;
    }
}

public sealed class MatrixFloatHandle : SafeHandle
{
    private MatrixFloatHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.matrix_free_float(handle);
        return true;
    }
}

public sealed class MatrixHalfHandle : SafeHandle
{
    private MatrixHalfHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        NativeMethods.matrix_free_half(handle);
        return true;
    }
}




public sealed class NativeMatrixDouble : IDisposable
{
    public MatrixDoubleHandle Handle { get; }
    public int Rows { get; }
    public int Cols { get; }

    public NativeMatrixDouble(int rows, int cols)
    {
        MatrixDoubleHandle testHandle = NativeMethods.matrix_double_create(rows, cols);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_double_create));
        if (testHandle.IsInvalid)
        {
            testHandle.Dispose();
            throw new OutOfMemoryException("Failed to allocate native matrix.");
        }
        Handle = testHandle;
        Rows = rows;
        Cols = cols;
    }

    private NativeMatrixDouble(MatrixDoubleHandle handle, int rows, int cols)
    {
        Handle = handle;
        Rows = rows;
        Cols = cols;
    }

    public void FillRandom(ulong seed)
    {
        NativeMethods.fill_double_rng(Handle, seed);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.fill_double_rng));
    }

    public void Fill(double value)
    {
        NativeMethods.fill_double_all_double(Handle, value);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.fill_double_all_double));
    }

    public double this[int r, int c] {
        get {
            if (r < 0 || r >= Rows || c < 0 || c >= Cols) throw new IndexOutOfRangeException($"Invalid indices r={r}, c={c}. Valid range is 0 <= r < {Rows}, 0 <= c < {Cols}");

            double value = NativeMethods.matrix_getdouble(Handle, r, c);
            NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_getdouble));
            return value;
        }
    }

    public static NativeMatrixDouble Multiply(NativeMatrixDouble a, NativeMatrixDouble b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Cols != b.Rows)
            throw new ArgumentException("Matrices are not compatible for multiplication.");

        MatrixDoubleHandle result = NativeMethods.matrix_multiply_double(
            a.Handle, b.Handle, a.Rows, a.Cols, b.Cols);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_multiply_double));
        return new NativeMatrixDouble(result, a.Rows, b.Cols);
    }

    public void Dispose() => Handle.Dispose();
}


public sealed class NativeMatrixFloat : IDisposable
{
    public MatrixFloatHandle Handle { get; }
    public int Rows { get; }
    public int Cols { get; }

    public NativeMatrixFloat(int rows, int cols)
    {
        //TODO: make sure the dimensions are not negative
        MatrixFloatHandle testHandle = NativeMethods.matrix_float_create(rows, cols);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_float_create));
        if (testHandle.IsInvalid)
        {
            testHandle.Dispose();
            throw new OutOfMemoryException("Failed to allocate native matrix.");
        }
        Handle = testHandle;
        Rows = rows;
        Cols = cols;
    }

    private NativeMatrixFloat(MatrixFloatHandle handle, int rows, int cols)
    {
        //TODO: make sure the dimensions are not negative
        Handle = handle;
        Rows = rows;
        Cols = cols;
    }
    public float this[int r, int c]
    {
        get
        {
            float value = NativeMethods.matrix_getfloat(Handle, r, c);
            NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_getfloat));
            return value;
        }
    }

    public void FillRandom(ulong seed)
    {
        NativeMethods.fill_float_rng(Handle, seed);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.fill_float_rng));
    }

    public int GetRows() {
        return Rows;
    }

    public int GetCols() {
        return Cols;
    }

    public static NativeMatrixFloat SumColumn(NativeMatrixFloat a)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        MatrixFloatHandle result = NativeMethods.column_sum_float(a.Handle);
        
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.column_sum_float));
        return new NativeMatrixFloat(result, 1, a.Cols);
    }


    /// <summary>
    /// Fill the matrix with values drawn from a Kaiming normal distribution.
    /// The memory resides on the GPU and is populated in-place.
    /// </summary>
    /// <param name="fanIn">Number of input units in the weight tensor.</param>
    /// <param name="stddev">Base standard deviation before fan-in adjustment.</param>
    /// <param name="seed">Random seed passed to cuRAND.</param>
    /// <param name="subseq">Optional subsequence for deterministic streams.</param>
    public void FillKaimingNormal(int fanIn, ulong seed, ulong subseq = 0)
    {
        NativeMethods.fill_kaiming_normal_float(Handle, fanIn, seed, subseq);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.fill_kaiming_normal_float));
    }

    public static NativeMatrixFloat LeakyReluFloat(NativeMatrixFloat a, float alpha)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.leaky_relu_float(a.Handle, alpha);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.leaky_relu_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat TanhFloat(NativeMatrixFloat a)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.tanh_float(a.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.tanh_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat DLeakyReluFloat(NativeMatrixFloat a, float alpha)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.d_leaky_relu_float(a.Handle, alpha);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.d_leaky_relu_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat ElementwiseMultiplyFloat(NativeMatrixFloat a, NativeMatrixFloat b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException("Matrices must have the same dimensions for elementwise multiplication.");

        MatrixFloatHandle result = NativeMethods.elementwise_multiplication_float(a.Handle, b.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.elementwise_multiplication_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat TransposeFloat(NativeMatrixFloat a)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.transpose_float(a.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.transpose_float));
        return new NativeMatrixFloat(result, a.Cols, a.Rows);
    }

    public static NativeMatrixFloat SigmoidFloat(NativeMatrixFloat a)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.sigmoid_float(a.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.sigmoid_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat ScalarDivisionFloat(NativeMatrixFloat a, float value)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.scalar_division_float(a.Handle, value);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.scalar_division_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat ScalarMultiplicationFloat(NativeMatrixFloat a, float value)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.scalar_multiplication_float(a.Handle, value);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.scalar_multiplication_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat ScalarAddFloat(NativeMatrixFloat a, float value)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.scalar_add_float(a.Handle, value);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.scalar_add_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat ScalarSubtractFloat(NativeMatrixFloat a, float value)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));

        MatrixFloatHandle result = NativeMethods.scalar_subtract_float(a.Handle, value);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.scalar_subtract_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat BroadcastRowwiseAddFloat(NativeMatrixFloat a, NativeMatrixFloat b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (b.Rows != 1 || b.Cols != a.Cols)
            throw new ArgumentException("b must be a row vector with the same number of columns as a.");

        MatrixFloatHandle result = NativeMethods.broadcast_rowwise_add_float(a.Handle, b.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.broadcast_rowwise_add_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat BroadcastRowwiseSubtractFloat(NativeMatrixFloat a, NativeMatrixFloat b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (b.Rows != 1 || b.Cols != a.Cols)
            throw new ArgumentException("b must be a row vector with the same number of columns as a.");

        MatrixFloatHandle result = NativeMethods.broadcast_rowwise_subtract_float(a.Handle, b.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.broadcast_rowwise_subtract_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat MatrixAddFloat(NativeMatrixFloat a, NativeMatrixFloat b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException("Matrices must have identical dimensions for addition.");

        MatrixFloatHandle result = NativeMethods.matrix_add_float(a.Handle, b.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_add_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat MatrixSubtractFloat(NativeMatrixFloat a, NativeMatrixFloat b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Rows != b.Rows || a.Cols != b.Cols)
            throw new ArgumentException("Matrices must have identical dimensions for subtraction.");

        MatrixFloatHandle result = NativeMethods.matrix_subtract_float(a.Handle, b.Handle);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_subtract_float));
        return new NativeMatrixFloat(result, a.Rows, a.Cols);
    }

    public static NativeMatrixFloat Multiply(NativeMatrixFloat a, NativeMatrixFloat b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Cols != b.Rows)
            throw new ArgumentException("Matrices are not compatible for multiplication.");

        MatrixFloatHandle result = NativeMethods.matrix_multiply_float(
            a.Handle, b.Handle, a.Rows, a.Cols, b.Cols);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_multiply_float));
        return new NativeMatrixFloat(result, a.Rows, b.Cols);
    }

    public void Dispose() => Handle.Dispose();
}



public sealed class NativeMatrixHalf : IDisposable
{
    public MatrixHalfHandle Handle { get; }
    public int Rows { get; }
    public int Cols { get; }

    public NativeMatrixHalf(int rows, int cols)
    {
        MatrixHalfHandle testHandle = NativeMethods.matrix_half_create(rows, cols);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_half_create));
        if (testHandle.IsInvalid)
        {
            testHandle.Dispose();
            throw new OutOfMemoryException("Failed to allocate native matrix.");
        }
        Handle = testHandle;
        Rows = rows;
        Cols = cols;
    }

    private NativeMatrixHalf(MatrixHalfHandle handle, int rows, int cols)
    {
        Handle = handle;
        Rows = rows;
        Cols = cols;
    }
    

    public void FillRandom(ulong seed, ulong subseq)
    {
        NativeMethods.fill_half_rng(Handle, seed, subseq);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.fill_half_rng));
    }

    public Half this[int r, int c]
    {
        get
        {
            // Convert raw ushort bits returned from native code into System.Half
            ushort bits = NativeMethods.matrix_gethalf(Handle, r, c);
            NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_gethalf));
            return BitConverter.UInt16BitsToHalf(bits);
        }
    }

    public static NativeMatrixHalf Multiply(NativeMatrixHalf a, NativeMatrixHalf b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Cols != b.Rows)
            throw new ArgumentException("Matrices are not compatible for multiplication.");

        MatrixHalfHandle result = NativeMethods.matrix_multiply_half(
            a.Handle, b.Handle, a.Rows, a.Cols, b.Cols);
        NativeMethods.ThrowIfFailed(nameof(NativeMethods.matrix_multiply_half));
        return new NativeMatrixHalf(result, a.Rows, b.Cols);
    }

    public void Dispose() => Handle.Dispose();
}