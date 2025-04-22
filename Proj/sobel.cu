#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

unsigned char* read_pgm(const char* filename, int* width, int* height) {
    
    // Read PGM file
    FILE* f = fopen(filename, "rb");
    if(!f) {
        perror("File open failed");
        exit(0);
    }

    // To check ASCII PGM (P2) or Binary PGM (P5)
    char magic[3];
    fscanf(f, "%2s", magic);

    // Accept only Binary PGM (P5)
    if(magic[0] != 'P' || magic[1] != '5') {
        printf("Unsupported Format: %s. Only P5", magic);
        fclose(f);
        exit(0);
    }

    // Obtain width, height and maxval from header after comments end
    int maxval;
    fscanf(f, "%d %d %d", width, height, &maxval);
    fgetc(f);       // Skip newline after header


    // Allocate memory for image based on height and width
    int size = *width * *height;
    unsigned char* data = (unsigned char*) malloc(size);

    // Read size amount of bytes from remaining PGM file
    fread(data, 1, size, f);
    fclose(f);

    return data;
}

__global__ void Conv2D(unsigned char* img, int* mask, unsigned char* res, int width, int height, int mw, int mh) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // Column
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // Row
    if(x < width && y < height) {
        // Get half width and height to perform convolution around target block
        int hmw = mw / 2;
        int hmh = mh / 2;
        int sum = 0;
        // i represents x mask coordinates
        for(int i = -hmw; i <= hmw; i++) {
            // j represents y mask coordinates
            for(int j = -hmh; j <= hmh; j++) {
                // xi represents x img coordinates
                int xi = x + i;
                // yi represents y img coordinates
                int yi = y + j;
                // Zero padding
                if(xi >= 0 && xi < width && yi >= 0 && yi < height) {
                    int img_val = img[yi * width + xi];
                    int mask_val = mask[(j + hmw) * mw + (i + hmh)];
                    sum += img_val * mask_val;
                }
            }
        }
        // Min: 0, Max: 255
        sum = min(max(sum, 0), 255);
        res[y * width + x] = (unsigned char) sum;
    }
}

int main() {

    const char* fname = "Mona_Lisa.pgm";

    int mask[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int width = 0, height = 0;
    unsigned char* img = read_pgm(fname, &width, &height);

    // Image and result same size
    int img_size = width * height * sizeof(unsigned char);
    int mask_size = 9 * sizeof(int);
    unsigned char* res = (unsigned char*) malloc(img_size);
    unsigned char *d_img, *d_res;
    int* d_mask;
    
    cudaMalloc((void**) &d_img, img_size);
    cudaMalloc((void**) &d_res, img_size);
    cudaMalloc((void**) &d_mask, mask_size);

    cudaMemcpy(d_img, img, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, mask_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    Conv2D<<<gridSize, blockSize>>> (d_img, d_mask, d_res, width, height, 3, 3);

    cudaEventRecord(stop);   
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Execution Time: %.4f ms\n", ms);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(res, d_res, img_size, cudaMemcpyDeviceToHost);

    FILE* out = fopen("Mona_Lisa_sobel_cu.pgm", "wb");
    fprintf(out, "P5\n%d %d\n255\n", width, height);
    fwrite(res, sizeof(unsigned char), img_size, out);
    fclose(out);
    printf("Saved filtered image as Mona_Lisa_sobel_cu.pgm\n");

    cudaFree(d_img);
    cudaFree(d_mask);
    cudaFree(d_res);
    free(img);
    free(res);
}