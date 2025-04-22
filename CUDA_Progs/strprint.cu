#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define MAX_COUNT 5

__global__ void func(int* a, char* c, char* res, int m, int n) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    int cid = blockIdx.y * blockDim.y + threadIdx.y;
    if(rid >= m || cid >= n) {
        return;
    }
    int idx = rid * n + cid;
    char ch = c[idx];
    int val = a[idx];
    int st = idx * MAX_COUNT;
    for(int i = 0; i < val; i++) {
        res[st++] = ch;
    }
    res[st] = '\0';
}

int main() {
    int *a, *da, m, n;
    char *c, *dc, *res, *dres;
    printf("\nEnter the row and column val:\n");
    scanf("%d%d", &m, &n);
    int a_size = m * n * sizeof(int);
    int c_size = m * n * sizeof(char);
    int res_size = MAX_COUNT * m * n * sizeof(char);
    a = (int*) malloc(a_size);
    c = (char*) malloc(c_size);
    res = (char*) malloc(res_size);
    for(int i = 0; i < m*n; i++) {
        a[i] = rand() % (MAX_COUNT-1) + 1;
        c[i] = rand() % 26 + ((rand()%2) ? 'A' : 'a'); 
    }
    printf("\nA:\n");
    for(int i = 0; i < m*n; i++) {
        printf("%d ", a[i]);
        if((i+1)%n == 0) {
            printf("\n");
        }
    }
    printf("\nC:\n");
    for(int i = 0; i < m*n; i++) {
        printf("%c ", c[i]);
        if((i+1)%n == 0) {
            printf("\n");
        }
    }
    
    cudaMalloc((void**) &da, a_size);
    cudaMalloc((void**) &dc, c_size);
    cudaMalloc((void**) &dres, res_size);
    cudaMemcpy(da, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, c_size, cudaMemcpyHostToDevice);
    dim3 dimBlock(4,4,1);
    dim3 dimGrid((m+3)/4, (n+3)/4, 1);
    func<<<dimGrid, dimBlock>>> (da, dc, dres, m, n);
    cudaMemcpy(res, dres, res_size, cudaMemcpyDeviceToHost);
    
    printf("\nResultant: ");
    for(int i = 0; i < m*n; i++) {
        printf("%s", res + i*MAX_COUNT);
    }
    free(a); free(c); free(res);
    cudaFree(da); cudaFree(dc); cudaFree(dres);
}