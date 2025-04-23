#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define KERNEL_WIDTH 3
#define MAXI 31
#define MINI 0
#define RRAND(min, max) \
    ( (float)(min) + ((float)(max) - (float)(min)) * ((float)rand() / (float)RAND_MAX) )

__constant__ float mask[KERNEL_WIDTH*KERNEL_WIDTH];

__global__ void conv2d(float* n, float* p, int h, int w) {
    int rid = blockIdx.y * blockDim.y + threadIdx.y;
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if(rid >= h || cid >= w) {
        return;
    }
    float val = 0.0;
    int h_st = rid - KERNEL_WIDTH/2;
    int w_st = cid - KERNEL_WIDTH/2;
    for(int i = 0; i < KERNEL_WIDTH; i++) {
        int h_i = h_st + i;
        for(int j = 0; j < KERNEL_WIDTH; j++) {
            int w_j = w_st + j;
            if(h_i >= 0 && h_i < h && w_j >= 0 && w_j < w) {
                val += n[h_i*w + w_j] * mask[i*KERNEL_WIDTH + j];
            }
        }
    }
    p[rid*w + cid] = val;
}

int main() {
    float *n, *m, *p, *dn, *dp;
    int h, w;
    printf("\nEnter the height and width:\n");
    scanf("%d%d", &h, &w);
    int n_size = h * w * sizeof(float);
    int m_size = KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float);
    n = (float*) malloc(n_size);
    p = (float*) malloc(n_size);
    m = (float*) malloc(m_size);
    printf("\nOriginal Matrix:\n");
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            n[i*w + j] = RRAND(MINI, MAXI);
            printf("%.2f ", n[i*w + j]);
        }
        printf("\n");
    }
    printf("\n\nMask:\n");
    for(int i = 0; i < KERNEL_WIDTH; i++) {
        for(int j = 0; j < KERNEL_WIDTH; j++) {
            m[i*KERNEL_WIDTH + j] = RRAND(MINI, MAXI);
            printf("%.2f ", m[i*KERNEL_WIDTH + j]);
        }
        printf("\n");
    }

    cudaMalloc((void**) &dn, n_size);
    cudaMalloc((void**) &dp, n_size);
    cudaMemcpy(dn, n, n_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, m, m_size);
    dim3 dimBlock(16,16);
    dim3 dimGrid((w+15)/16, (h+15)/16);
    conv2d<<<dimGrid, dimBlock>>> (dn, dp, h, w);
    cudaDeviceSynchronize();
    cudaMemcpy(p, dp, n_size, cudaMemcpyDeviceToHost);

    printf("\nResultant:\n");
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            printf("%.2f ",p[i*w + j]);
        }
        printf("\n");
    }
    free(n); free(m); free(p);
    cudaFree(dn); cudaFree(mask); cudaFree(dp);
}