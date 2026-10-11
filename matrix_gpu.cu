#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

// 检查 CUDA 操作，失败时打印原因并退出。
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n",
                     operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

// 基础 CUDA 矩阵乘法：每个线程计算一个输出元素。
__global__ void matrixMultiplyGPU(
    const float *A, const float *B, float *C, int N) {

    // 根据线程块位置和块内编号，确定负责的行、列。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 跳过越界线程，将 A 的一行与 B 的一列相乘求和。
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char **argv) {
    // 读取矩阵边长，默认 512，允许范围为 1～4096。
    long value = 512;
    if (argc > 1) {
        char *end;
        value = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' ||
            value < 1 || value > 4096) {
            std::fprintf(stderr, "N must be from 1 to 4096\n");
            return 1;
        }
    }

    // 在 CPU 上创建输入和输出矩阵，按行存放。
    int N = static_cast<int>(value);
    size_t count = static_cast<size_t>(N) * N;
    size_t bytes = count * sizeof(float);
    std::vector<float> A(count), B(count), C(count);

    // 固定随机种子，生成输入数据。
    std::srand(42);
    for (size_t i = 0; i < count; ++i) {
        A[i] = (std::rand() % 100) / 100.0f;
        B[i] = (std::rand() % 100) / 100.0f;
    }

    // 分配 GPU 显存，并上传输入矩阵。
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    check(cudaMalloc((void **)&d_A, bytes), "Allocate A");
    check(cudaMalloc((void **)&d_B, bytes), "Allocate B");
    check(cudaMalloc((void **)&d_C, bytes), "Allocate C");

    check(cudaMemcpy(d_A, A.data(), bytes,
                     cudaMemcpyHostToDevice), "Upload A");
    check(cudaMemcpy(d_B, B.data(), bytes,
                     cudaMemcpyHostToDevice), "Upload B");

    // 每块 16×16 个线程，网格向上取整以覆盖整个矩阵。
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // 先预热一次，不计入正式测量。
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
    check(cudaGetLastError(), "Warm-up launch");
    check(cudaDeviceSynchronize(), "Warm-up execution");

    // 用 CUDA 事件计时，不包含显存分配和数据传输。
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "Create start event");
    check(cudaEventCreate(&stop), "Create stop event");

    // 测量三次，最后取中位数。
    float times[3];
    for (int run = 0; run < 3; ++run) {
        check(cudaEventRecord(start), "Record start");

        matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);
        check(cudaGetLastError(), "Kernel launch");

        check(cudaEventRecord(stop), "Record stop");
        check(cudaEventSynchronize(stop), "Wait for GPU");
        check(cudaEventElapsedTime(&times[run], start, stop),
              "Measure elapsed time");

        std::printf("N=%d run=%d kernel_ms=%.6f\n",
                    N, run + 1, times[run]);
    }

    // 将 GPU 计算结果复制回 CPU。
    check(cudaMemcpy(C.data(), d_C, bytes,
                     cudaMemcpyDeviceToHost), "Download C");

    // CPU 逐元素计算参考答案，不计入 GPU 耗时。
    std::printf("Checking every element against CPU...\n");
    std::fflush(stdout);

    size_t errors = 0;
    double maxAbsError = 0.0;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float reference = 0.0f;
            for (int k = 0; k < N; ++k) {
                reference += A[row * N + k] * B[k * N + col];
            }

            // 允许少量浮点误差，并检查 NaN 或无穷大。
            float actual = C[row * N + col];
            double error = std::fabs(
                static_cast<double>(actual) - reference);
            double tolerance = 1e-4 + 1e-4 * std::fabs(reference);

            if (!std::isfinite(actual) || error > tolerance) {
                ++errors;
            }
            maxAbsError = std::max(maxAbsError, error);
        }
    }

    // 输出中位耗时、最大误差和验证结果。
    std::sort(times, times + 3);
    std::printf("Median kernel time (N=%d): %.6f ms\n", N, times[1]);
    std::printf("Max absolute error: %.8g\n", maxAbsError);
    std::printf("Verification: %s (%zu mismatches)\n",
                errors == 0 ? "PASSED" : "FAILED", errors);

    // 释放计时事件和 GPU 显存。
    check(cudaEventDestroy(start), "Destroy start event");
    check(cudaEventDestroy(stop), "Destroy stop event");
    check(cudaFree(d_A), "Free A");
    check(cudaFree(d_B), "Free B");
    check(cudaFree(d_C), "Free C");

    return errors == 0 ? 0 : 1;
}
