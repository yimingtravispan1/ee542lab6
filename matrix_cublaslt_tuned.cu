#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <limits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// 检查 CUDA 操作，失败时打印原因并退出。
void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n",
                     operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

// 检查 cuBLAS 操作是否成功。
void checkBlas(cublasStatus_t status, const char *operation)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s: cuBLAS error %d\n",
                     operation, static_cast<int>(status));
        std::exit(1);
    }
}

// 将耗时排序并取中位数，本程序传入的测量次数均为奇数。
float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

int main(int argc, char **argv)
{
    // 读取矩阵边长，默认 512，允许范围为 1～4096。
    long value = 512;

    if (argc > 1) {
        char *end = nullptr;
        value = std::strtol(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0' ||
            value < 1 || value > 4096) {
            std::fprintf(stderr, "N must be from 1 to 4096\n");
            return 1;
        }
    }

    const int N = static_cast<int>(value);
    const size_t count = static_cast<size_t>(N) * N;
    const size_t bytes = count * sizeof(float);

    // 两种实现使用相同输入，分别保存结果用于检查。
    std::vector<float> A(count), B(count);
    std::vector<float> baselineResult(count), tunedResult(count);

    std::srand(42);
    for (size_t i = 0; i < count; ++i) {
        A[i] = (std::rand() % 100) / 100.0f;
        B[i] = (std::rand() % 100) / 100.0f;
    }

    // 分配 GPU 显存，上传输入并清空输出。
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    check(cudaMalloc((void **)&d_A, bytes), "Allocate A");
    check(cudaMalloc((void **)&d_B, bytes), "Allocate B");
    check(cudaMalloc((void **)&d_C, bytes), "Allocate C");

    check(cudaMemcpy(d_A, A.data(), bytes,
                     cudaMemcpyHostToDevice), "Upload A");
    check(cudaMemcpy(d_B, B.data(), bytes,
                     cudaMemcpyHostToDevice), "Upload B");
    check(cudaMemset(d_C, 0, bytes), "Initialize C");

    // 普通 cuBLAS 作为性能比较的基准。
    cublasHandle_t handle;
    checkBlas(cublasCreate(&handle), "Create cuBLAS");
    checkBlas(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH),
              "Set baseline math mode");

    // 创建 cuBLASLt，并指定使用单精度浮点计算。
    cublasLtHandle_t lt;
    checkBlas(cublasLtCreate(&lt), "Create cuBLASLt");

    cublasLtMatmulDesc_t operation;
    checkBlas(
        cublasLtMatmulDescCreate(
            &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "Create operation descriptor");

    // 两个输入矩阵都不额外转置。
    cublasOperation_t noTranspose = CUBLAS_OP_N;
    checkBlas(
        cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSA,
            &noTranspose, sizeof(noTranspose)),
        "Set transpose A");
    checkBlas(
        cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &noTranspose, sizeof(noTranspose)),
        "Set transpose B");

    // 描述 N×N 矩阵的存储布局，默认按列主序解释。
    cublasLtMatrixLayout_t layout;
    checkBlas(
        cublasLtMatrixLayoutCreate(
            &layout, CUDA_R_32F, N, N, N),
        "Create matrix layout");

    // 给候选算法提供最多 32 MiB 的额外工作空间。
    const size_t workspaceBytes = 32ULL * 1024 * 1024;
    void *workspace = nullptr;
    check(cudaMalloc(&workspace, workspaceBytes),
          "Allocate workspace");

    cublasLtMatmulPreference_t preference;
    checkBlas(cublasLtMatmulPreferenceCreate(&preference),
              "Create preference");
    checkBlas(
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspaceBytes, sizeof(workspaceBytes)),
        "Set workspace limit");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 封装普通 cuBLAS 调用，方便重复运行。
    // 交换 A、B 的传入顺序，使行主序结果仍表示 A×B。
    auto runBaseline = [&]() {
        return cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha, d_B, N, d_A, N,
            &beta, d_C, N);
    };

    // 使用指定的 cuBLASLt 算法计算相同的矩阵乘法。
    auto runLt = [&](const cublasLtMatmulAlgo_t *algo) {
        return cublasLtMatmul(
            lt, operation,
            &alpha,
            d_B, layout,
            d_A, layout,
            &beta,
            d_C, layout,
            d_C, layout,
            algo,
            workspace, workspaceBytes,
            0);
    };

    // 用 CUDA 事件测量 GPU 计算时间，不包含输入输出传输。
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "Create start event");
    check(cudaEventCreate(&stop), "Create stop event");

    auto measure = [&](auto launch) {
        check(cudaEventRecord(start), "Record start");
        checkBlas(launch(), "Timed GEMM");
        check(cudaEventRecord(stop), "Record stop");
        check(cudaEventSynchronize(stop), "Wait for GEMM");

        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop),
              "Read elapsed time");
        return ms;
    };

    std::printf("N=%d\n", N);
    std::printf("Baseline: cuBLAS SGEMM, DEFAULT math\n");
    std::printf("Tuned: cuBLASLt, CUBLAS_COMPUTE_32F\n");
    std::printf("Workspace limit: %zu bytes\n", workspaceBytes);

    // 正式搜索前先预热 GPU。
    for (int i = 0; i < 5; ++i) {
        checkBlas(runBaseline(), "Initial warm-up");
    }
    check(cudaDeviceSynchronize(), "Finish initial warm-up");

    // 单独记录算法搜索耗时，避免将调优成本隐藏在结果之外。
    auto tuningStart = std::chrono::steady_clock::now();

    // 向 cuBLASLt 请求最多 32 个候选算法。
    constexpr int MAX_CANDIDATES = 32;
    cublasLtMatmulHeuristicResult_t candidates[MAX_CANDIDATES] = {};
    int returned = 0;

    checkBlas(
        cublasLtMatmulAlgoGetHeuristic(
            lt, operation,
            layout, layout, layout, layout,
            preference,
            MAX_CANDIDATES, candidates, &returned),
        "Query candidate algorithms");

    std::printf("Returned candidates: %d\n", returned);

    int best = -1;
    float bestSearchMs = std::numeric_limits<float>::infinity();

    // 每个可用候选算法先预热，再测量五次。
    for (int i = 0; i < returned; ++i) {
        if (candidates[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }

        cublasStatus_t status = runLt(&candidates[i].algo);

        if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
            std::printf("candidate=%d skipped: not supported\n", i);
            continue;
        }

        checkBlas(status, "Candidate warm-up");
        check(cudaDeviceSynchronize(), "Finish candidate warm-up");

        std::vector<float> searchTimes;
        for (int run = 0; run < 5; ++run) {
            searchTimes.push_back(measure([&]() {
                return runLt(&candidates[i].algo);
            }));
        }

        float ms = median(searchTimes);

        std::printf(
            "candidate=%d search_median_ms=%.6f workspace=%zu\n",
            i, ms, candidates[i].workspaceSize);

        // 保留测得中位耗时最短的候选算法。
        if (ms < bestSearchMs) {
            bestSearchMs = ms;
            best = i;
        }
    }

    auto tuningEnd = std::chrono::steady_clock::now();
    double tuningMs =
        std::chrono::duration<double, std::milli>(
            tuningEnd - tuningStart).count();

    if (best < 0) {
        std::fprintf(stderr, "No usable cuBLASLt candidate found\n");
        return 1;
    }

    std::printf("Selected candidate index: %d\n", best);
    std::printf("Selected workspace requirement: %zu bytes\n",
                candidates[best].workspaceSize);
    std::printf("Search wall time: %.3f ms\n", tuningMs);

    // 搜索结束后，重新预热两种实现，再做独立的性能比较。
    for (int i = 0; i < 5; ++i) {
        checkBlas(runBaseline(), "Baseline warm-up");
        checkBlas(runLt(&candidates[best].algo), "Tuned warm-up");
    }
    check(cudaDeviceSynchronize(), "Finish comparison warm-up");

    std::vector<float> baselineTimes;
    std::vector<float> tunedTimes;

    // 比较七轮，交替执行顺序，减少固定先后顺序带来的偏差。
    for (int run = 0; run < 7; ++run) {
        float baselineMs;
        float tunedMs;

        if (run % 2 == 0) {
            baselineMs = measure(runBaseline);
            tunedMs = measure([&]() {
                return runLt(&candidates[best].algo);
            });
        } else {
            tunedMs = measure([&]() {
                return runLt(&candidates[best].algo);
            });
            baselineMs = measure(runBaseline);
        }

        baselineTimes.push_back(baselineMs);
        tunedTimes.push_back(tunedMs);

        std::printf(
            "run=%d baseline_ms=%.6f tuned_ms=%.6f\n",
            run + 1, baselineMs, tunedMs);
    }

    // 分别取回两种实现的结果，用于正确性检查。
    checkBlas(runBaseline(), "Baseline validation run");
    check(cudaMemcpy(baselineResult.data(), d_C, bytes,
                     cudaMemcpyDeviceToHost),
          "Download baseline result");

    checkBlas(runLt(&candidates[best].algo), "Tuned validation run");
    check(cudaMemcpy(tunedResult.data(), d_C, bytes,
                     cudaMemcpyDeviceToHost),
          "Download tuned result");

    std::printf("Checking both results against CPU...\n");
    std::fflush(stdout);

    size_t baselineErrors = 0;
    size_t tunedErrors = 0;
    double baselineMaxError = 0.0;
    double tunedMaxError = 0.0;

    // CPU 逐元素计算参考答案，同时检查两种 GPU 实现。
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float reference = 0.0f;

            for (int k = 0; k < N; ++k) {
                reference += A[row * N + k] * B[k * N + col];
            }

            // 浮点计算允许少量误差，不要求结果完全相等。
            size_t index = static_cast<size_t>(row) * N + col;
            double tolerance = 1e-4 + 1e-4 * std::fabs(reference);

            double e0 = std::fabs(
                static_cast<double>(baselineResult[index]) - reference);
            double e1 = std::fabs(
                static_cast<double>(tunedResult[index]) - reference);

            if (!std::isfinite(baselineResult[index]) || e0 > tolerance)
                ++baselineErrors;

            if (!std::isfinite(tunedResult[index]) || e1 > tolerance)
                ++tunedErrors;

            baselineMaxError = std::max(baselineMaxError, e0);
            tunedMaxError = std::max(tunedMaxError, e1);
        }
    }

    // 输出中位耗时与加速比；加速比大于 1 才表示调优版更快。
    float baselineMedian = median(baselineTimes);
    float tunedMedian = median(tunedTimes);

    std::printf("Median cuBLAS baseline: %.6f ms\n", baselineMedian);
    std::printf("Median tuned cuBLASLt: %.6f ms\n", tunedMedian);
    std::printf("Speedup baseline/tuned: %.4fx\n",
                baselineMedian / tunedMedian);

    std::printf("Baseline max error: %.8g\n", baselineMaxError);
    std::printf("Tuned max error: %.8g\n", tunedMaxError);

    std::printf("Baseline verification: %s (%zu mismatches)\n",
                baselineErrors == 0 ? "PASSED" : "FAILED",
                baselineErrors);
    std::printf("Tuned verification: %s (%zu mismatches)\n",
                tunedErrors == 0 ? "PASSED" : "FAILED",
                tunedErrors);

    // 释放计时事件、库对象和 GPU 显存。
    check(cudaEventDestroy(start), "Destroy start event");
    check(cudaEventDestroy(stop), "Destroy stop event");
    checkBlas(cublasLtMatmulPreferenceDestroy(preference),
              "Destroy preference");
    checkBlas(cublasLtMatrixLayoutDestroy(layout), "Destroy layout");
    checkBlas(cublasLtMatmulDescDestroy(operation), "Destroy operation");
    checkBlas(cublasLtDestroy(lt), "Destroy cuBLASLt");
    checkBlas(cublasDestroy(handle), "Destroy cuBLAS");

    check(cudaFree(workspace), "Free workspace");
    check(cudaFree(d_A), "Free A");
    check(cudaFree(d_B), "Free B");
    check(cudaFree(d_C), "Free C");

    return (baselineErrors == 0 && tunedErrors == 0) ? 0 : 1;
}
