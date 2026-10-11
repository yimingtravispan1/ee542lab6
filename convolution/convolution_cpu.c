#include "convolution.h"
#include <stddef.h>

int conv2d_cpu(
    const uint32_t *image,
    const float *filter,
    float *output,
    int M,
    int N)
{


    if (!image || !filter || !output ||
        M < 1 || M > 8192 ||
        N < 1 || N > 31 || N > M ||
        N % 2 == 0) {
        return -1;
    }


    int radius = N / 2;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < M; ++col) {

            float sum = 0.0f;
            for (int ky = 0; ky < N; ++ky) {
                for (int kx = 0; kx < N; ++kx) {
                    int imageRow = row + ky - radius;
                    int imageCol = col + kx - radius;
                    if (imageRow < 0 || imageRow >= M ||
                        imageCol < 0 || imageCol >= M) {
                        continue;
                    }

                    int filterRow = N - 1 - ky;
                    int filterCol = N - 1 - kx;

                    size_t imageIndex =
                        (size_t)imageRow * M + imageCol;

                    size_t filterIndex =
                        (size_t)filterRow * N + filterCol;
                    sum += (float)image[imageIndex]
                         * filter[filterIndex];
                }
            }

            output[(size_t)row * M + col] = sum;
        }
    }

    return 0;
}
