#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define KERNEL_SIZE 2
#define STRIDE 2

__global__ void average_pooling_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // output x
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // output y

    int out_width = width / STRIDE;
    int out_height = height / STRIDE;

    if (x < out_width && y < out_height) {
        int sum = 0;
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                int in_x = x * STRIDE + j;
                int in_y = y * STRIDE + i;
                sum += input[in_y * width + in_x];
            }
        }
        output[y * out_width + x] = sum / (KERNEL_SIZE * KERNEL_SIZE);
    }
}

void read_pgm(const char* filename, unsigned char** data, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("File open failed"); exit(1); }

    char header[3];
    fscanf(f, "%2s", header);
    if (header[0] != 'P' || header[1] != '5') {
        fprintf(stderr, "Unsupported file format\n");
        exit(1);
    }

    fscanf(f, "%d %d", width, height);
    int maxval;
    fscanf(f, "%d", &maxval);
    fgetc(f); // skip newline

    *data = (unsigned char*)malloc(*width * *height);
    fread(*data, 1, *width * *height, f);
    fclose(f);
}

void write_pgm(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height, f);
    fclose(f);
}

int main() {
    const char* input_file = "Mona_Lisa.pgm";
    const char* output_file = "output.pgm";

    unsigned char *h_input, *h_output;
    unsigned char *d_input, *d_output;
    int width, height;

    // Read input image
    read_pgm(input_file, &h_input, &width, &height);

    int out_width = width / STRIDE;
    int out_height = height / STRIDE;

    h_output = (unsigned char*)malloc(out_width * out_height);

    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, out_width * out_height);

    cudaMemcpy(d_input, h_input, width * height, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    dim3 blockSize(16, 16);
    dim3 gridSize((out_width + 15) / 16, (out_height + 15) / 16);
    average_pooling_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);   
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Execution Time: %.4f ms\n", ms);

    cudaMemcpy(h_output, d_output, out_width * out_height, cudaMemcpyDeviceToHost);

    write_pgm(output_file, h_output, out_width, out_height);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    printf("Average pooling complete. Output saved to %s\n", output_file);
    return 0;
}
