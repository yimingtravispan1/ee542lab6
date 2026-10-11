#include "convolution_gpu.h"
#include <cuda_runtime.h>
#include <stddef.h>

__global__ static void convolution_kernel(const uint32_t *image,
                                          const float *filter, float *output,
                                          int M, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= M) return;
    float sum = 0.0f;
    for (int ky = 0; ky < N; ++ky) {
        for (int kx = 0; kx < N; ++kx) {
            int r = row + ky - N / 2;
            int c = col + kx - N / 2;
            if (r >= 0 && r < M && c >= 0 && c < M)
                sum += (float)image[(size_t)r * M + c]
                     * filter[(N - 1 - ky) * N + N - 1 - kx];
        }
    }
    output[(size_t)row * M + col] = sum;
}

extern "C" int gpu_convolution_timed(const uint32_t *image, const float *filter,
                                     float *output, int M, int N, float *kernel_ms)
{
    if (kernel_ms) *kernel_ms = 0.0f;
    if (!image || !filter || !output || M < 1 || M > 8192 ||
        N < 1 || N > 31 || N > M || N % 2 == 0)
        return (int)cudaErrorInvalidValue;
    uint32_t *d_image = nullptr;
    float *d_filter = nullptr, *d_output = nullptr;
    cudaEvent_t start = nullptr, end = nullptr;
    cudaError_t status = cudaSuccess;
    const size_t pixels = (size_t)M * M;
    dim3 block(16, 16), grid((M + 15) / 16, (M + 15) / 16);
#define CHECK(call) do { status = (call); if (status != cudaSuccess) goto cleanup; } while (0)
    CHECK(cudaMalloc((void **)&d_image, pixels * sizeof(uint32_t)));
    CHECK(cudaMalloc((void **)&d_filter, (size_t)N * N * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_output, pixels * sizeof(float)));
    CHECK(cudaMemcpy(d_image, image, pixels * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_filter, filter, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
    if (kernel_ms) {
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&end));
        CHECK(cudaEventRecord(start));
    }
    convolution_kernel<<<grid, block>>>(d_image, d_filter, d_output, M, N);
    CHECK(cudaGetLastError());
    if (kernel_ms) {
        CHECK(cudaEventRecord(end));
        CHECK(cudaEventSynchronize(end));
        CHECK(cudaEventElapsedTime(kernel_ms, start, end));
    } else {
        CHECK(cudaDeviceSynchronize());
    }
    CHECK(cudaMemcpy(output, d_output, pixels * sizeof(float), cudaMemcpyDeviceToHost));
cleanup:
    // Preserve the first failure, including failures during cleanup.
#define RELEASE(call) do { cudaError_t s = (call); if (status == cudaSuccess) status = s; } while (0)
    if (end) RELEASE(cudaEventDestroy(end));
    if (start) RELEASE(cudaEventDestroy(start));
    if (d_output) RELEASE(cudaFree(d_output));
    if (d_filter) RELEASE(cudaFree(d_filter));
    if (d_image) RELEASE(cudaFree(d_image));
#undef RELEASE
#undef CHECK
    return (int)status;
}

extern "C" int gpu_convolution(const uint32_t *image, const float *filter,
                               float *output, int M, int N)
{
    return gpu_convolution_timed(image, filter, output, M, N, nullptr);
}
