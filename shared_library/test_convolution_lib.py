"""Run on the GPU server after bash convolution/build_part8.sh."""
import ctypes
from pathlib import Path

import numpy as np

from convolution_lib import ConvolutionLibrary


def check(name, actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    assert np.isfinite(actual).all(), name
    print(f"{name}: PASSED; max_abs_error={np.max(np.abs(actual.astype(np.float64) - expected)):.8g}")


def main():
    gpu = ConvolutionLibrary()
    cpu = ctypes.CDLL(str(Path(__file__).resolve().parent / "libconvolution_cpu.so"))
    cpu.conv2d_cpu.argtypes = gpu.lib.gpu_convolution.argtypes
    cpu.conv2d_cpu.restype = ctypes.c_int
    image = np.arange(1, 10, dtype=np.uint32).reshape(3, 3)
    identity = np.zeros((3, 3), dtype=np.float32)
    identity[1, 1] = 1
    check("Identity", gpu.convolve(image, identity), image)
    asymmetric = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    expected = np.array([[26, 56, 54], [84, 165, 144], [134, 236, 186]])
    check("Asymmetric and zero padding", gpu.convolve(image, asymmetric), expected)
    profiled, kernel_ms = gpu.convolve(image, asymmetric, profile=True)
    check("Profiled interface", profiled, expected)
    assert np.isfinite(kernel_ms) and kernel_ms >= 0
    impulse = np.zeros((3, 3), dtype=np.uint32)
    impulse[1, 1] = 1
    edge = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    check("Signed output", gpu.convolve(impulse, edge), edge)
    rng = np.random.default_rng(542)
    for m, n in [(1, 1), (17, 3), (65, 5), (65, 31), (128, 7)]:
        image = rng.integers(0, 256, (m, m), dtype=np.uint32)
        kernel = rng.random((n, n), dtype=np.float32) / (n * n)
        expected = np.empty((m, m), dtype=np.float32)
        assert cpu.conv2d_cpu(image.ravel(), kernel.ravel(), expected.ravel(), m, n) == 0
        check(f"CPU/GPU M={m}, N={n}", gpu.convolve(image, kernel), expected)
    # Noncontiguous input and values above the 8-bit image range.
    image = np.array([[0, 2**32 - 1], [65536, 123456]], dtype=np.uint32).T
    check("uint32 range and layout", gpu.convolve(image, np.ones((1, 1))), image.astype(np.float32))
    for image, kernel in [(np.ones((3, 4)), identity), (impulse, np.ones((2, 2))),
                          (-np.ones((3, 3)), identity), (impulse, np.full((3, 3), np.nan))]:
        try:
            gpu.convolve(image, kernel)
        except ValueError:
            continue
        raise AssertionError("Invalid Python input accepted")
    # Raw C API must reject invalid dimensions as well.
    image = impulse.ravel()
    output = np.empty(9, dtype=np.float32)
    for m, n in [(0, 3), (3, 2), (3, 5), (8193, 3), (3, 0)]:
        assert gpu.lib.gpu_convolution(image, edge.ravel(), output, m, n) != 0
    raw_call = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_int, ctypes.c_int)(
        ctypes.cast(gpu.lib.gpu_convolution, ctypes.c_void_p).value)
    addresses = [image.ctypes.data, edge.ctypes.data, output.ctypes.data]
    for index in range(3):
        pointers = addresses.copy()
        pointers[index] = None
        assert raw_call(*pointers, 3, 3) != 0
    print("All Part 8 convolution tests PASSED.")


if __name__ == "__main__":
    main()
