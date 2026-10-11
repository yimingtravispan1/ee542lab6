import ctypes
from pathlib import Path
from time import perf_counter

import numpy as np


# 从当前脚本所在目录加载 CUDA 共享库。
library_path = Path(__file__).resolve().parent / "libmatrix.so"
lib = ctypes.CDLL(str(library_path))

# 输入必须是一维 float32 数组，内存连续且对齐。
input_array = np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=1,
    flags=("C_CONTIGUOUS", "ALIGNED"),
)

# 输出数组还需要允许写入。
output_array = np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=1,
    flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE"),
)

# 声明 C 接口的参数类型和返回类型。
lib.gpu_matrix_multiply.argtypes = [
    input_array,
    input_array,
    output_array,
    ctypes.c_int,
]
lib.gpu_matrix_multiply.restype = ctypes.c_int

# 错误查询接口接收整数错误码，返回字符串。
lib.gpu_error_string.argtypes = [ctypes.c_int]
lib.gpu_error_string.restype = ctypes.c_char_p


def gpu_matmul(a, b):
    """Multiply two square matrices of the same size and return a float32 array."""

    # 统一数据类型和内存布局，满足 C 接口要求。
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)

    # 检查输入是否为尺寸相同的方阵。
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("A and B must be two-dimensional arrays")

    if a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")

    n = a.shape[0]
    if not 1 <= n <= 4096:
        raise ValueError("N must be between 1 and 4096")

    c = np.zeros((n, n), dtype=np.float32)

    # 将矩阵展平后传入共享库，结果直接写入 c。
    status = lib.gpu_matrix_multiply(
        a.ravel(), b.ravel(), c.ravel(), n
    )

    # 将 CUDA 错误转换为 Python 异常。
    if status != 0:
        message = lib.gpu_error_string(status).decode("utf-8")
        raise RuntimeError(f"CUDA error {status}: {message}")

    return c


def check_case(name, a, b, expected=None):
    # 测量整个封装调用，包括数据准备、显存分配、传输和计算。
    start = perf_counter()
    actual = gpu_matmul(a, b)
    elapsed_ms = (perf_counter() - start) * 1000.0

    # 未提供已知答案时，用 float64 计算参考结果。
    # 先转为 float32，保证参考计算与 GPU 使用相同的输入值。
    if expected is None:
        a32 = np.asarray(a, dtype=np.float32)
        b32 = np.asarray(b, dtype=np.float32)
        expected = a32.astype(np.float64) @ b32.astype(np.float64)

    # 逐元素比较，允许少量浮点误差，并排除 NaN 和无穷大。
    error = np.abs(actual.astype(np.float64) - expected)
    tolerance = 1e-4 + 1e-4 * np.abs(expected)

    valid = np.isfinite(actual) & (error <= tolerance)
    mismatches = int(np.count_nonzero(~valid))

    print(f"{name}: shape={actual.shape}, dtype={actual.dtype}")
    print(f"  Python wrapper elapsed: {elapsed_ms:.3f} ms")
    print(f"  Max absolute error: {error.max():.8g}")
    print(
        f"  Verification: {'PASSED' if mismatches == 0 else 'FAILED'} "
        f"({mismatches} mismatches)"
    )

    if mismatches:
        raise AssertionError(f"{name} failed correctness verification")


if __name__ == "__main__":
    # 用已知答案的 2×2 矩阵检查基本计算。
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
    check_case("Known 2x2", a, b, expected)

    rng = np.random.default_rng(42)

    # 65 不是分块大小 16 的整数倍，用于检查边界处理。
    a = rng.random((65, 65)).astype(np.float32)
    b = rng.random((65, 65)).astype(np.float32)
    check_case("Boundary N=65", a, b)

    # 使用较大矩阵做一次基础验证。
    a = rng.random((512, 512)).astype(np.float32)
    b = rng.random((512, 512)).astype(np.float32)
    check_case("Basic N=512", a, b)

    print("All Part 7 basic tests PASSED.")
