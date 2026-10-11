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
#define ROWS_PER_THREAD 4

// 每个线程计算同一列的 4 个输出元素，复用读取的 B 数据。
__global__ void matrixMultiplyRegister(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    // 共享内存暂存 A 的 64×16 分块和 B 的 16×16 分块。
    __shared__ float tileA[TILE_WIDTH * ROWS_PER_THREAD][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int blockRow = blockIdx.y * TILE_WIDTH * ROWS_PER_THREAD;

    // 每个线程保存 4 个累加结果，供编译器尽量放入寄存器。
    float sums[ROWS_PER_THREAD] = {0.0f};

    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t)
    {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        // 每个线程加载 A 中的 4 个元素，超出矩阵范围时补 0。
        // unroll 提示编译器展开循环，减少循环控制开销。
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
        {
            int localRow = ty + r * TILE_WIDTH;
            int row = blockRow + localRow;

            tileA[localRow][tx] =
                (row < N && aCol < N)
                ? A[row * N + aCol]
                : 0.0f;
        }

        // 每个线程加载 B 中的一个元素。
        tileB[ty][tx] =
            (bRow < N && col < N)
            ? B[bRow * N + col]
            : 0.0f;

        // 等待整个线程块完成加载，再使用共享数据。
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            // 同一个 B 元素用于更新 4 个输出，减少重复读取。
            float b = tileB[k][tx];

            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r)
            {
                sums[r] += tileA[ty + r * TILE_WIDTH][k] * b;
            }
        }

        // 等所有线程用完当前分块，再加载下一个分块。
        __syncthreads();
    }

    // 将 4 个结果写回 GPU 显存，只写入矩阵范围内的位置。
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r)
    {
        int row = blockRow + ty + r * TILE_WIDTH;

        if (row < N && col < N)
        {
            C[row * N + col] = sums[r];
        }
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

    // 固定随机种子，生成输入矩阵。
    std::srand(42);
    for (size_t i = 0; i < count; ++i) {
        A[i] = (std::rand() % 100) / 100.0f;
        B[i] = (std::rand() % 100) / 100.0f;
    }

    // 分配 GPU 显存，并上传输入。
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    check(cudaMalloc((void **)&d_A, bytes), "Allocate A");
    check(cudaMalloc((void **)&d_B, bytes), "Allocate B");
    check(cudaMalloc((void **)&d_C, bytes), "Allocate C");

    check(cudaMemcpy(d_A, A.data(), bytes,
                     cudaMemcpyHostToDevice), "Upload A");
    check(cudaMemcpy(d_B, B.data(), bytes,
                     cudaMemcpyHostToDevice), "Upload B");

    // 每块有 16×16 个线程，共负责 64 行、16 列的输出。
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH * ROWS_PER_THREAD - 1)
            / (TILE_WIDTH * ROWS_PER_THREAD)
    );

    // 先预热一次，不计入正式测量。
    matrixMultiplyRegister<<<grid, block>>>(d_A, d_B, d_C, N);
    check(cudaGetLastError(), "Warm-up launch");
    check(cudaDeviceSynchronize(), "Warm-up execution");

    // 使用 CUDA 事件计时，不包含显存分配和数据传输。
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "Create start event");
    check(cudaEventCreate(&stop), "Create stop event");

    float times[3];
    for (int run = 0; run < 3; ++run) {
        check(cudaEventRecord(start), "Record start");

        matrixMultiplyRegister<<<grid, block>>>(d_A, d_B, d_C, N);
        check(cudaGetLastError(), "Kernel launch");

        check(cudaEventRecord(stop), "Record stop");
        check(cudaEventSynchronize(stop), "Wait for GPU");
        check(cudaEventElapsedTime(&times[run], start, stop),
              "Measure elapsed time");

        std::printf("N=%d run=%d kernel_ms=%.6f\n",
                    N, run + 1, times[run]);
    }

    // 取回 GPU 结果，准备与 CPU 计算结果比较。
    check(cudaMemcpy(C.data(), d_C, bytes,
                     cudaMemcpyDeviceToHost), "Download C");

    std::printf("Checking every element against CPU...\n");
    std::fflush(stdout);

    // 逐元素验证，允许少量浮点误差。
    size_t errors = 0;
    double maxAbsError = 0.0;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float reference = 0.0f;
            for (int k = 0; k < N; ++k) {
                reference += A[row * N + k] * B[k * N + col];
            }

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

    // 输出三次耗时的中位数，以及正确性检查结果。
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
