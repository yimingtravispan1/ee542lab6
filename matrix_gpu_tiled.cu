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

#define TILE_WIDTH 16

// 分块矩阵乘法：每个线程计算一个输出，共享输入数据以减少显存读取。
__global__ void matrixMultiplyTiled(
    const float *A, const float *B, float *C, int N) {

    // 线程块共用的两个 16×16 数据分块。
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 当前线程负责的输出行、列。
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // 沿乘法的求和方向，逐块加载并计算。
    for (int m = 0;
         m < (N + TILE_WIDTH - 1) / TILE_WIDTH;
         ++m) {

        // 每个线程加载一个 A 元素，越界位置补 0。
        if (Row < N && m * TILE_WIDTH + tx < N) {
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        // 每个线程加载一个 B 元素，越界位置补 0。
        if (Col < N && m * TILE_WIDTH + ty < N) {
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        // 等整个线程块完成加载，再读取共享数据。
        __syncthreads();

        // 将当前分块的乘加结果累加到当前输出。
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        // 等所有线程用完当前数据，再加载下一块。
        __syncthreads();
    }

    // 只写入矩阵范围内的结果。
    if (Row < N && Col < N) {
        C[Row * N + Col] = Pvalue;
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

    int N = static_cast<int>(value);
    size_t count = static_cast<size_t>(N) * N;
    size_t bytes = count * sizeof(float);
    std::vector<float> A(count), B(count), C(count);

    // 使用与 CPU 版本相同的随机种子和初始化方式。
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

    // 每块 16×16 个线程，网格向上取整以覆盖整个输出矩阵。
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // 先预热一次，不计入正式测量。
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
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

        matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
        check(cudaGetLastError(), "Kernel launch");

        check(cudaEventRecord(stop), "Record stop");
        check(cudaEventSynchronize(stop), "Wait for GPU");
        check(cudaEventElapsedTime(&times[run], start, stop),
              "Measure elapsed time");

        std::printf("N=%d run=%d kernel_ms=%.6f\n",
                    N, run + 1, times[run]);
    }

    // 将 GPU 结果取回 CPU。
    check(cudaMemcpy(C.data(), d_C, bytes,
                     cudaMemcpyDeviceToHost), "Download C");

    // 在 CPU 上逐元素计算参考答案，这部分不计入 GPU 耗时。
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
