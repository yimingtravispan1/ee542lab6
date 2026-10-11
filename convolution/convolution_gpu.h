#ifndef CONVOLUTION_GPU_H
#define CONVOLUTION_GPU_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Host arrays; zero on success, otherwise a CUDA error code.
 * Same convolution, dimensions and types as conv2d_cpu.
 * Each call allocates, transfers, synchronizes and frees device buffers. */
int gpu_convolution(const uint32_t *image, const float *filter,
                    float *output, int M, int N);
/* Optional kernel-only CUDA-event time, excluding allocation and transfers. */
int gpu_convolution_timed(const uint32_t *image, const float *filter,
                          float *output, int M, int N, float *kernel_ms);
#ifdef __cplusplus
}
#endif
#endif
