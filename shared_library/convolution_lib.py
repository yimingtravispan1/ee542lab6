"""ctypes interface to the Part 8 CUDA convolution in libmatrix.so."""
import ctypes
from pathlib import Path

import numpy as np


class ConvolutionLibrary:
    def __init__(self, path=None):
        path = path or Path(__file__).resolve().parent / "libmatrix.so"
        self.lib = ctypes.CDLL(str(path))
        image = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1,
                                      flags=("C_CONTIGUOUS", "ALIGNED"))
        values = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                       flags=("C_CONTIGUOUS", "ALIGNED"))
        output = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1,
                                       flags=("C_CONTIGUOUS", "ALIGNED", "WRITEABLE"))
        args = [image, values, output, ctypes.c_int, ctypes.c_int]
        self.lib.gpu_convolution.argtypes = args
        self.lib.gpu_convolution.restype = ctypes.c_int
        self.lib.gpu_convolution_timed.argtypes = args + [ctypes.POINTER(ctypes.c_float)]
        self.lib.gpu_convolution_timed.restype = ctypes.c_int
        self.lib.gpu_error_string.argtypes = [ctypes.c_int]
        self.lib.gpu_error_string.restype = ctypes.c_char_p

    def convolve(self, image, kernel, *, profile=False):
        """Return float32 convolution (and kernel ms when profile=True).

        Preserve uint32 intensity values; reject negative/fractional inputs.
        Output allocation and input layout conversion are part of this wrapper.
        """
        image = np.asarray(image)
        kernel = np.asarray(kernel)
        if image.ndim != 2 or image.shape[0] != image.shape[1]:
            raise ValueError("image must be square")
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("filter must be square")
        m, n = image.shape[0], kernel.shape[0]
        if not (1 <= m <= 8192 and 1 <= n <= min(m, 31) and n % 2):
            raise ValueError("M must be 1..8192; N must be odd, 1..31 and <= M")
        if image.dtype != np.uint32:
            if (image.dtype.kind not in "uif" or not np.isfinite(image).all()
                    or np.any(image < 0) or np.any(image > np.iinfo(np.uint32).max)
                    or np.any(image != np.floor(image))):
                raise ValueError("pixels must be integers in the uint32 range")
        image = np.ascontiguousarray(image, dtype=np.uint32)
        kernel = np.ascontiguousarray(kernel, dtype=np.float32)
        if not np.isfinite(kernel).all():
            raise ValueError("filter must contain finite float32 coefficients")
        output = np.empty((m, m), dtype=np.float32)
        args = (image.ravel(), kernel.ravel(), output.ravel(), m, n)
        if profile:
            kernel_ms = ctypes.c_float()
            status = self.lib.gpu_convolution_timed(*args, ctypes.byref(kernel_ms))
        else:
            status = self.lib.gpu_convolution(*args)
        if status:
            message = self.lib.gpu_error_string(status).decode("utf-8")
            raise RuntimeError(f"CUDA error {status}: {message}")
        return (output, kernel_ms.value) if profile else output
