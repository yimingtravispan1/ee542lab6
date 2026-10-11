# ee542lab6
lab 6 workspace

Part 8.2 / 8.3 requires a Linux NVIDIA GPU server with CUDA Toolkit, GCC,
and Python packages `numpy`, `pillow`, and `matplotlib`.

Run from the repository root:

```bash
bash convolution/build_part8.sh
python3 shared_library/test_convolution_lib.py
python3 convolution/benchmark_part8.py \
  --inputs images/input_512.u32 images/input_1024.u32 images/input_2048.u32 \
  --filter-sizes 3 5 7 --repeats 7 --plot
```

Use the same `.u32` inputs and filter sizes as Part 8.1; the paths and sizes above
are examples. Results, correctness logs, and plots are saved in `results/part8/`.
Both executables reuse `convolution/convolution_main.c` for image I/O.
The shared library retains matrix multiplication and adds CUDA convolution.

Measurements exclude image I/O, process startup, and one warmup call. CPU timing
covers the convolution function; CUDA call timing includes device allocation,
transfers, synchronization, cleanup, and event management. Python timing also
includes wrapper validation, layout conversion, and output allocation. Kernel
event timings are reported separately. All timings are medians in milliseconds.
