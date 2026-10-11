#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "convolution.h"


int main(int argc, char *argv[])
{
    if (argc != 5) {
        fprintf(stderr,
                "Usage: %s input.u32 output.f32 "
                "mean|edge|sharpen N\n",
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

  
    if (conv2d_cpu(image, filter, output, M, N) != 0) {
        fprintf(stderr, "CPU convolution failed.\n");
        goto cleanup;
    }


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

    printf("CPU convolution completed.\n");
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
