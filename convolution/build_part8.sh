#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
gcc -O3 -std=c11 -D_POSIX_C_SOURCE=200809L convolution/convolution_main.c convolution/convolution_cpu.c -o build/convolution_cpu
gcc -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -DCONVOLUTION_GPU -c convolution/convolution_main.c -o build/convolution_gpu_main.o
nvcc -O3 -c convolution/convolution_gpu.cu -o build/convolution_gpu.o
nvcc build/convolution_gpu_main.o build/convolution_gpu.o -o build/convolution_gpu
gcc -O3 -shared -fPIC convolution/convolution_cpu.c -o shared_library/libconvolution_cpu.so
nvcc -O3 -shared -Xcompiler -fPIC shared_library/matrix_lib.cu convolution/convolution_gpu.cu -o shared_library/libmatrix.so
gcc -O3 -std=c11 convolution/test_convolution_cpu.c convolution/convolution_cpu.c -lm -o build/test_convolution_cpu
build/test_convolution_cpu
