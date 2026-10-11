#include <cuda_runtime.h>
#include <cstddef>

#define TILE_WIDTH 16

// 分块矩阵乘法：每个线程计算一个输出，共享输入数据。
__global__ void matrixMultiplyTiled(
    const float *A, const float *B, float *C, int N)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 当前线程负责的输出位置。
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_WIDTH + tx;
        int bRow = t * TILE_WIDTH + ty;

        // 将 A、B 的当前分块放入共享内存，越界位置补 0。
        tileA[ty][tx] =
            (row < N && aCol < N)
            ? A[row * N + aCol] : 0.0f;

        tileB[ty][tx] =
            (bRow < N && col < N)
            ? B[bRow * N + col] : 0.0f;

        // 等待整个线程块加载完成，再进行乘加。
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // 等所有线程用完当前数据，再加载下一块。
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA 操作失败时跳到清理步骤，释放已分配的资源。
#define CUDA_CHECK_CALL(expression)                  \
    do {                                             \
        status = (expression);                       \
        if (status != cudaSuccess) goto cleanup;     \
    } while (0)

// 导出 C 接口，方便 Python ctypes 按函数名调用。
// 输入和输出都是 CPU 数组，GPU 内存和数据传输由函数内部管理。
extern "C" int gpu_matrix_multiply(
    const float *h_A, const float *h_B, float *h_C, int N)
{
    // 检查数组指针和矩阵尺寸。
    if (!h_A || !h_B || !h_C || N < 1 || N > 4096) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    const size_t bytes =
        static_cast<size_t>(N) * N * sizeof(float);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    cudaError_t status = cudaSuccess;

    // 每块 16×16 个线程，网格覆盖整个输出矩阵。
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // 分配 GPU 显存。
    CUDA_CHECK_CALL(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK_CALL(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK_CALL(cudaMalloc((void **)&d_C, bytes));

    // 将 CPU 输入复制到 GPU。
    CUDA_CHECK_CALL(cudaMemcpy(
        d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_CALL(cudaMemcpy(
        d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 启动内核，检查启动错误并等待计算完成。
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);

    CUDA_CHECK_CALL(cudaGetLastError());
    CUDA_CHECK_CALL(cudaDeviceSynchronize());

    // 将结果写回调用者提供的 CPU 输出数组。
    CUDA_CHECK_CALL(cudaMemcpy(
        h_C, d_C, bytes, cudaMemcpyDeviceToHost));

cleanup:
    // 成功或失败都会释放显存，保留最先出现的错误。
    if (d_C) {
        cudaError_t s = cudaFree(d_C);
        if (status == cudaSuccess) status = s;
    }
    if (d_B) {
        cudaError_t s = cudaFree(d_B);
        if (status == cudaSuccess) status = s;
    }
    if (d_A) {
        cudaError_t s = cudaFree(d_A);
        if (status == cudaSuccess) status = s;
    }

    // 返回 0 表示成功，其他值为 CUDA 错误码，由调用者处理。
    return static_cast<int>(status);
}

#undef CUDA_CHECK_CALL

// 将错误码转换成文字说明，方便 Python 显示错误原因。
extern "C" const char *gpu_error_string(int code)
{
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}