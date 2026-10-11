#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdint.h>

/* 让后续 CUDA/C++ 代码也能调用这个 C 函数。 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * image：M×M 的灰度图片，像素类型为 uint32_t。
 * filter：N×N 的浮点滤波器。
 * output：M×M 的浮点输出，调用者提前分配。
 *
 * 输入与输出使用不同的数组。
 * 返回 0 表示成功，-1 表示参数无效。
 */
int conv2d_cpu(
    const uint32_t *image,
    const float *filter,
    float *output,
    int M,
    int N
);

#ifdef __cplusplus
}
#endif

#endif
