#include <stdio.h>
#include <math.h>
#include "convolution.h"

// 逐元素比较实际结果和已知答案，允许少量浮点误差。
static int check_result(
    const char *name,
    const float *actual,
    const float *expected,
    int count)
{
    int errors = 0;

    for (int i = 0; i < count; ++i) {
        double error = fabs((double)actual[i] - expected[i]);
        double tolerance = 1e-4 + 1e-4 * fabs(expected[i]);

        // 遇到非有限值或误差过大时，打印对应位置。
        if (!isfinite(actual[i]) || error > tolerance) {
            printf("  index=%d actual=%.6f expected=%.6f\n",
                   i, actual[i], expected[i]);
            ++errors;
        }
    }

    printf("%s: %s (%d mismatches)\n",
           name,
           errors == 0 ? "PASSED" : "FAILED",
           errors);

    // 通过返回 0，失败返回 1，方便累计失败的测试数。
    return errors != 0;
}

int main(void)
{
    int failures = 0;
    float output[9];

    // 用一个简单的 3×3 图片进行测试。
    const uint32_t image[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    // 测试 1：只有中心系数为 1，输出应与输入完全相同。
    const float identity[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };

    const float expected_identity[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    if (conv2d_cpu(image, identity, output, 3, 3) != 0) {
        fprintf(stderr, "Identity call failed\n");
        return 1;
    }

    failures += check_result(
        "Identity", output, expected_identity, 9);

    // 测试 2：用非对称滤波器检查滤波器翻转和边界补零。
    const float asymmetric[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    const float expected_asymmetric[9] = {
         26,  56,  54,
         84, 165, 144,
        134, 236, 186
    };

    if (conv2d_cpu(image, asymmetric, output, 3, 3) != 0) {
        fprintf(stderr, "Asymmetric call failed\n");
        return 1;
    }

    failures += check_result(
        "Asymmetric convolution",
        output, expected_asymmetric, 9);

    // 测试 3：对中心亮点做边缘检测，检查负数结果是否保留。
    const uint32_t impulse[9] = {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };

    const float edge[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };

    const float expected_edge[9] = {
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0
    };

    if (conv2d_cpu(impulse, edge, output, 3, 3) != 0) {
        fprintf(stderr, "Edge call failed\n");
        return 1;
    }

    failures += check_result(
        "Signed edge response",
        output, expected_edge, 9);

    // 汇总三个测试，全部通过才返回成功。
    printf("CPU convolution basic tests: %s\n",
           failures == 0 ? "PASSED" : "FAILED");

    return failures == 0 ? 0 : 1;
}