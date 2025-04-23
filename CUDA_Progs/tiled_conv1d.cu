#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define TILE_WIDTH 16
#define KERNEL_WIDTH 5
#define HALO (KERNEL_WIDTH/2)
#define MINI 0
#define MAXI 31
#define rrand(min, max) ((rand()%(max-min)) + min)

__constant__ float mask[KERNEL_WIDTH];

__global__ void conv1d(float* n, float* p, int w) {
    int idx = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // represents the extent to which the block will need neighboring data for conv
    __shared__ float sdata[TILE_WIDTH + 2*HALO];
    int base = blockIdx.x * TILE_WIDTH - HALO;

    // two loads for HALO threads, other than that useless for loop
    for(int i = threadIdx.x; i < TILE_WIDTH + 2 * HALO; i += TILE_WIDTH) {
        int ind = base + i;
        sdata[i] = (ind < 0 || ind >= w) ? 0.0f : n[ind];
    }
    // always sync after loading operation
    __syncthreads();
    if(idx < w) {
        float sum = 0.0;
        int center = threadIdx.x + HALO;
        for(int i = 0; i < KERNEL_WIDTH; i++) {
            sum += sdata[center + i - HALO] * mask[i];
        }
        p[idx] = sum;
    }

}

int main() {
    float *n, *m, *p, *dn, *dp;
    int w;
    printf("\nEnter the width:\n");
    scanf("%d", &w);
    int n_size = w * sizeof(float);
    int m_size = KERNEL_WIDTH * sizeof(float);
    n = (float*) malloc(n_size);
    m = (float*) malloc(m_size);
    p = (float*) malloc(n_size);
    printf("\nN:\n");
    for(int i = 0; i < w; i++) {
        n[i] = rrand(MINI, MAXI);
        printf("%.2f ", n[i]);
    }
    printf("\nM:\n");
    for(int i = 0; i < KERNEL_WIDTH; i++) {
        m[i] = rrand(MINI, MAXI);
        printf("%.2f ", m[i]);
    }
    cudaMalloc((void**)&dn, n_size);
    cudaMalloc((void**)&dp, n_size);
    cudaMemcpy(dn, n, n_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, m, m_size);
    dim3 dimBlock(TILE_WIDTH);
    dim3 dimGrid((w+TILE_WIDTH-1)/TILE_WIDTH);
    conv1d<<<dimGrid, dimBlock>>> (dn, dp, w);
    cudaDeviceSynchronize();
    cudaMemcpy(p, dp, n_size, cudaMemcpyDeviceToHost);
    printf("\nResult:\n");
    for(int i = 0; i < w; i++) {
        printf("%.2f ", p[i]);
    }
    free(n); free(m); free(p);
    cudaFree(dn); cudaFree(mask); cudaFree(dp);
}