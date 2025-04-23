#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>

__constant__ char key[30];

__global__ void find(char* sent, int* ind, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= len) {
        return;
    }
    int flag = 1, i;
    for(i = 0; sent[i + idx*30] != '\0' || key[i] != '\0'; i++) {
        if(sent[i + idx*30] != key[i]) {
            flag = 0;
            break;
        }
    }
    if(sent[i + idx*30] == '\0' && key[i] == '\0' && flag) {
        atomicMin(ind, idx);
    }
}

int main() {
    printf("\nEnter the number of words:\n");
    int len; scanf("%d", &len);
    char sentence[len][30], search[30];
    printf("\nEnter the sentence:\n");
    for(int i = 0; i < len; i++) {
        scanf("%s", sentence[i]);
    }
    printf("\nEnter the word to search:\n");
    scanf("%s", search);
    int s_size = len * 30 * sizeof(char);
    int ind = INT_MAX;
    int* dind; char *dsent;
    cudaMalloc((void**)&dsent, s_size);
    cudaMalloc((void**)&dind, sizeof(int));
    cudaMemcpy(dsent, sentence, s_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dind, &ind, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(key, search, 30 * sizeof(char));
    dim3 dimBlock(16);
    dim3 dimGrid((len+15)/16);
    find<<<dimGrid, dimBlock>>>(dsent, dind, len);
    cudaMemcpy(&ind, dind, sizeof(int), cudaMemcpyDeviceToHost);
    printf("\n\nIndex: %d", ind);
    cudaFree(dsent); cudaFree(dind); cudaFree(key);
}