#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 检查 CUDA 操作，失败时打印原因并退出。
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n",
                     operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

// 检查 cuBLAS 操作，失败时打印错误码。
void checkBlas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLAS error %d\n",
                     operation, static_cast<int>(status));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    // 从命令行读取矩阵边长，默认 512，允许范围为 1～4096。
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

    // 在 CPU 上创建三个矩阵，vector 会自动管理内存。
    int N = static_cast<int>(value);
    size_t count = static_cast<size_t>(N) * N;
    size_t bytes = count * sizeof(float);
    std::vector<float> A(count), B(count), C(count);

    // 固定随机种子，生成用于测试的输入矩阵。
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

    // 创建 cuBLAS 运行上下文，并设置默认数学模式。
    cublasHandle_t handle;
    checkBlas(cublasCreate(&handle), "Create cuBLAS handle");

    checkBlas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH),
              "Set cuBLAS math mode");

    std::printf("Implementation: cuBLAS SGEMM; math mode: DEFAULT\n");

    // SGEMM 计算 C = alpha × A × B + beta × C，这里只保留 A × B。
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 先预热一次，不计入正式测量。
    // cuBLAS 使用列主序，因此交换 A、B 的传入顺序，
    // 让结果按本程序的行主序读取时仍然是 A × B。
    checkBlas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N),
        "Warm-up SGEMM");

    check(cudaDeviceSynchronize(), "Warm-up execution");

    // 使用 CUDA 事件计时，不包含显存分配和 CPU/GPU 数据传输。
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "Create start event");
    check(cudaEventCreate(&stop), "Create stop event");

    // 测量三次，最后取中位数，减少偶然波动的影响。
    float times[3];
    for (int run = 0; run < 3; ++run) {
        check(cudaEventRecord(start), "Record start");

        checkBlas(
            cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N),
            "Timed SGEMM");

        check(cudaEventRecord(stop), "Record stop");
        check(cudaEventSynchronize(stop), "Wait for GPU");
        check(cudaEventElapsedTime(&times[run], start, stop),
              "Measure elapsed time");

        std::printf("N=%d run=%d kernel_ms=%.6f\n",
                    N, run + 1, times[run]);
    }

    // 将最后一次计算结果取回 CPU。
    check(cudaMemcpy(C.data(), d_C, bytes,
                     cudaMemcpyDeviceToHost), "Download C");

    std::printf("Checking every element against CPU...\n");
    std::fflush(stdout);

    // 在 CPU 上重新计算每个元素，与 GPU 结果逐一比较。
    size_t errors = 0;
    double maxAbsError = 0.0;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float reference = 0.0f;
            for (int k = 0; k < N; ++k) {
                reference += A[row * N + k] * B[k * N + col];
            }

            // 允许少量浮点误差，并检查结果是否包含 NaN 或无穷大。
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

    // 输出三次耗时的中位数、最大误差和验证结果。
    std::sort(times, times + 3);
    std::printf("Median cuBLAS GPU time (N=%d): %.6f ms\n", N, times[1]);
    std::printf("Max absolute error: %.8g\n", maxAbsError);
    std::printf("Verification: %s (%zu mismatches)\n",
                errors == 0 ? "PASSED" : "FAILED", errors);

    // 释放计时事件、cuBLAS 上下文和 GPU 显存。
    check(cudaEventDestroy(start), "Destroy start event");
    check(cudaEventDestroy(stop), "Destroy stop event");
    checkBlas(cublasDestroy(handle), "Destroy cuBLAS handle");
    check(cudaFree(d_A), "Free A");
    check(cudaFree(d_B), "Free B");
    check(cudaFree(d_C), "Free C");

    return errors == 0 ? 0 : 1;
}
