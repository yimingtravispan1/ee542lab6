#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 检查 CUDA 操作，失败时打印原因并退出。
void check(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n",
                     operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

// GPU 上每个线程负责一个元素，将它加 10。
__global__ void addTen(int *data) {
    int i = threadIdx.x;
    data[i] += 10;
}

int main() {
    // 获取并显示第一块 GPU 的名称。
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0), "Get GPU information");
    std::printf("GPU: %s\n", prop.name);

    // 准备 CPU 数据，device 用来保存 GPU 显存地址。
    int host[4] = {1, 2, 3, 4};
    int *device = nullptr;

    // 分配 GPU 显存，并把输入数据复制过去。
    check(cudaMalloc((void **)&device, sizeof(host)), "Allocate GPU memory");
    check(cudaMemcpy(device, host, sizeof(host),
                     cudaMemcpyHostToDevice), "Copy CPU to GPU");

    // 启动 1 个线程块、4 个线程，检查启动错误并等待计算完成。
    addTen<<<1, 4>>>(device);
    check(cudaGetLastError(), "Launch kernel");
    check(cudaDeviceSynchronize(), "Execute kernel");

    // 把结果取回 CPU，然后释放 GPU 显存。
    check(cudaMemcpy(host, device, sizeof(host),
                     cudaMemcpyDeviceToHost), "Copy GPU to CPU");
    check(cudaFree(device), "Free GPU memory");

    std::printf("Result: %d %d %d %d\n",
                host[0], host[1], host[2], host[3]);

    // 检查结果是否为 11、12、13、14。
    for (int i = 0; i < 4; ++i) {
        if (host[i] != i + 11) {
            std::printf("CUDA test FAILED\n");
            return 1;
        }
    }

    std::printf("CUDA test PASSED\n");
    return 0;
}
