#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CPU 矩阵乘法：A 的第 i 行与 B 的第 j 列相乘求和，得到 C[i][j]。
void matrixMultiplyCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    // 先用已知答案的 2×2 矩阵检查基本计算是否正确。
    float testA[4] = {1, 2, 3, 4};
    float testB[4] = {5, 6, 7, 8};
    float testC[4];
    matrixMultiplyCPU(testA, testB, testC, 2);

    if (testC[0] != 19 || testC[1] != 22 ||
        testC[2] != 43 || testC[3] != 50) {
        fprintf(stderr, "Correctness check FAILED\n");
        return 1;
    }
    printf("Correctness check PASSED\n");

    // 从命令行读取矩阵边长，默认 1024，允许范围为 1～8192。
    long inputN = 1024;
    if (argc > 1) {
        char *end;
        inputN = strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' ||
            inputN < 1 || inputN > 8192) {
            fprintf(stderr, "N must be an integer from 1 to 8192\n");
            return 1;
        }
    }
    int N = (int)inputN;
    size_t count = (size_t)N * N;
    size_t bytes = count * sizeof(float);

    // 为三个 N×N 矩阵分配内存，按行存放在一维数组中。
    float *A = malloc(bytes);
    float *B = malloc(bytes);
    float *C = malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    // 固定随机种子，方便在相同环境下重复生成相同输入。
    srand(42);
    for (size_t i = 0; i < count; i++) {
        A[i] = (rand() % 100) / 100.0f;
        B[i] = (rand() % 100) / 100.0f;
    }

    // 只统计矩阵乘法的 CPU 时间，不包含内存分配和数据准备。
    clock_t start = clock();
    matrixMultiplyCPU(A, B, C, N);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    // 将结果求和作为简单摘要，不能代替逐元素正确性检查。
    double checksum = 0.0;
    for (size_t i = 0; i < count; i++) {
        checksum += C[i];
    }

    printf("CPU execution time (N=%d): %.6f seconds\n", N, elapsed);
    printf("Checksum: %.6f\n", checksum);

    // 释放内存。
    free(A);
    free(B);
    free(C);
    return 0;
}