#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

__global__ void conv1d(int* n, int* m, int* p, int nn, int nm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= 0 && idx < nn) {
        int start = idx - nm/2;
        int val = 0;
        for(int i = 0; i < nm; i++) {
            if(start + i >= 0 && start + i < nn) {
                val += n[start+i] * m[i];
            }
        }
        p[idx] = val;
    }
}

int main() {
    int *n, *m, *p, *dn, *dm, *dp;
    printf("\nEnter the width of N and M respectively:\n");
    int nn, nm; scanf("%d%d", &nn, &nm);
    int n_size = nn * sizeof(int), m_size = nm * sizeof(int);
    n = (int*) malloc(n_size);
    m = (int*) malloc(m_size);
    p = (int*) malloc(n_size);
    for(int i = 0; i < nn; i++) {
        n[i] = rand() % 256;
    }
    for(int i = 0; i < nm; i++) {
        m[i] = rand() % 256;
    }
    printf("\nN: ");
    for(int i = 0; i < nn; i++) {
        printf("%d ", n[i]);
    }
    printf("\nM: ");
    for(int i = 0; i < nm; i++) {
        printf("%d ", m[i]);
    }
    printf("\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void**)&dn, n_size);
    cudaMalloc((void**)&dm, m_size);
    cudaMalloc((void**)&dp, n_size);
    cudaMemcpy(dn, n, n_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dm, m, m_size, cudaMemcpyHostToDevice);
    dim3 dimBlock(64,1,1);
    dim3 dimGrid((nn+64-1)/64,1,1);
    cudaEventRecord(start, 0);
    conv1d<<<dimGrid,dimBlock>>> (dn, dm, dp, nn, nm);
    cudaEventRecord(stop, 0);
    cudaMemcpy(p, dp, n_size, cudaMemcpyDeviceToHost);
    printf("\nResult:\n");
    for(int i = 0; i < nn; i++) {
        printf("%d ", p[i]);
    }
    float el_time;
    cudaEventElapsedTime(&el_time, start, stop);
    printf("\n\nElapsed Time: %.4f ms", el_time);
    free(p); free(m); free(n);
    cudaFree(dp); cudaFree(dm); cudaFree(dn);
    return 0;
}
