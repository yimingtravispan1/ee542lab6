#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "convolution.h"
#ifdef CONVOLUTION_GPU
#include "convolution_gpu.h"
#endif

static double elapsed_ms(struct timespec start, struct timespec end)
{
    return (end.tv_sec - start.tv_sec) * 1000.0
         + (end.tv_nsec - start.tv_nsec) / 1000000.0;
}

static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}


int main(int argc, char *argv[])
{
    if (argc != 5 && argc != 6) {
        fprintf(stderr,
                "Usage: %s input.u32 output.f32 "
                "mean|edge|sharpen N [repeats]\n",
                argv[0]);
        return 1;
    }

    uint32_t endian_check = 1;
    if (*(const unsigned char *)&endian_check != 1 ||
        sizeof(float) != 4) {
        fprintf(stderr,
                "This program requires little-endian "
                "and 32-bit float.\n");
        return 1;
    }

    char *end = NULL;
    long parsed_n = strtol(argv[4], &end, 10);

    if (end == argv[4] || *end != '\0' ||
        parsed_n < 1 || parsed_n > 31 ||
        parsed_n % 2 == 0) {
        fprintf(stderr,
                "N must be an odd integer from 1 to 31.\n");
        return 1;
    }

    int N = (int)parsed_n;
    int repeats = 7;
    if (argc == 6) {
        long value = strtol(argv[5], &end, 10);
        if (end == argv[5] || *end != '\0' || value < 1 || value > 1000) {
            fprintf(stderr, "repeats must be between 1 and 1000.\n");
            return 1;
        }
        repeats = (int)value;
    }
    float filter[31 * 31] = {0};

    if (strcmp(argv[3], "mean") == 0) {
        for (int i = 0; i < N * N; ++i)
            filter[i] = 1.0f / (float)(N * N);

    } else if (strcmp(argv[3], "edge") == 0 ||
               strcmp(argv[3], "sharpen") == 0) {

        if (N != 3) {
            fprintf(stderr,
                    "edge and sharpen require N=3.\n");
            return 1;
        }

        const float kernel[9] = {
             0, -1,  0,
            -1,  4, -1,
             0, -1,  0
        };

        memcpy(filter, kernel, sizeof(kernel));

        if (strcmp(argv[3], "sharpen") == 0)
            filter[4] = 5.0f;

    } else {
        fprintf(stderr, "Unknown filter: %s\n", argv[3]);
        return 1;
    }

    FILE *input_file = NULL;
    FILE *output_file = NULL;
    uint32_t *image = NULL;
    float *output = NULL;
    int status = 1;


    input_file = fopen(argv[1], "rb");
    if (!input_file) {
        perror("Open input");
        goto cleanup;
    }

    char header[64];
    char extra;
    int M = 0;

    if (!fgets(header, sizeof(header), input_file) ||
        !strchr(header, '\n') ||
        sscanf(header, "U32 %d %c", &M, &extra) != 1 ||
        M < 1 || M > 8192 || N > M) {
        fprintf(stderr, "Invalid image header or dimensions.\n");
        goto cleanup;
    }

    size_t count = (size_t)M * (size_t)M;

    image = malloc(count * sizeof(*image));
    output = malloc(count * sizeof(*output));

    if (!image || !output) {
        fprintf(stderr, "Memory allocation failed.\n");
        goto cleanup;
    }

   
    if (fread(image, sizeof(*image), count, input_file) != count) {
        fprintf(stderr, "Input pixel data is incomplete.\n");
        goto cleanup;
    }

    if (fgetc(input_file) != EOF || ferror(input_file)) {
        fprintf(stderr, "Extra input data or read error.\n");
        goto cleanup;
    }

    fclose(input_file);
    input_file = NULL;

  
    /* One warmup; time only convolution calls, excluding file I/O and
     * host buffer allocation. GPU calls include allocation and transfers. */
    double call_times[1000], kernel_times[1000];
    for (int run = -1; run < repeats; ++run) {
        struct timespec begin, finish;
        float kernel_ms = 0.0f;
        if (clock_gettime(CLOCK_MONOTONIC, &begin) != 0) goto cleanup;
#ifdef CONVOLUTION_GPU
        int code = gpu_convolution_timed(image, filter, output, M, N, &kernel_ms);
#else
        int code = conv2d_cpu(image, filter, output, M, N);
#endif
        if (clock_gettime(CLOCK_MONOTONIC, &finish) != 0) goto cleanup;
        if (code != 0) {
            fprintf(stderr, "Convolution failed: %d\n", code);
            goto cleanup;
        }
        if (run >= 0) {
            call_times[run] = elapsed_ms(begin, finish);
            kernel_times[run] = kernel_ms;
        }
    }
    qsort(call_times, repeats, sizeof(double), compare_double);
    qsort(kernel_times, repeats, sizeof(double), compare_double);
    printf("call_median_ms=%.9f\n", (call_times[(repeats - 1) / 2] + call_times[repeats / 2]) / 2);
#ifdef CONVOLUTION_GPU
    printf("kernel_median_ms=%.9f\n", (kernel_times[(repeats - 1) / 2] + kernel_times[repeats / 2]) / 2);
#endif


    output_file = fopen(argv[2], "wb");
    if (!output_file) {
        perror("Open output");
        goto cleanup;
    }

    if (fprintf(output_file, "F32 %d\n", M) < 0 ||
        fwrite(output, sizeof(*output), count, output_file)
            != count) {
        fprintf(stderr, "Write output failed.\n");
        goto cleanup;
    }

   
    int close_result = fclose(output_file);
    output_file = NULL;

    if (close_result != 0) {
        perror("Close output");
        goto cleanup;
    }

#ifdef CONVOLUTION_GPU
    printf("GPU convolution completed.\n");
#else
    printf("CPU convolution completed.\n");
#endif
    printf("Image: %d x %d\n", M, M);
    printf("Filter: %s, %d x %d\n", argv[3], N, N);
    printf("Output: %s\n", argv[2]);

    status = 0;

cleanup:
    if (input_file)
        fclose(input_file);
    if (output_file)
        fclose(output_file);

    free(image);
    free(output);
    return status;
}
