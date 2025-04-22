#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <string.h>
#define MAX_WORD_SIZE 50

__device__ void swap(char* a, char* b) {
    char temp = *a;
    *a = *b;
    *b = temp;
}

__global__ void rev(char* sent, int num_words) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= 0 && idx < num_words) {
        char* s = sent + MAX_WORD_SIZE * idx;
        int len = 0;
        for(int i = 0; s[i] != '\0'; i++) {
            len++;
        }
        for(int i = 0; i < len/2; i++) {
            swap(&s[i], &s[len-i-1]);
        }
    }
}

int main() {
    int num_words = 0, sent_size = 0;
    printf("\nEnter the number of words:\n");
    scanf("%d", &num_words);
    char inter[num_words][50];
    for(int i = 0; i < num_words; i++) {
        scanf("%s", inter[i]);
    }
    sent_size = num_words * MAX_WORD_SIZE * sizeof(char);
    char* sent = (char*) malloc(sent_size);
    for(int i = 0; i < num_words; i++) {
        strcpy(sent+i*MAX_WORD_SIZE, inter[i]);
    }
    char *dsent;
    cudaMalloc((void**) &dsent, sent_size);
    cudaMemcpy(dsent, sent, sent_size, cudaMemcpyHostToDevice);
    dim3 dimBlock(64,1,1);
    dim3 dimGrid((num_words + 63)/64, 1, 1);
    rev<<<dimGrid, dimBlock>>> (dsent, num_words);
    cudaMemcpy(sent, dsent, sent_size, cudaMemcpyDeviceToHost);
    printf("\nNew string:\n");
    for(int i = 0; i < num_words; i++) {
        printf("%s ", sent+i*MAX_WORD_SIZE);
    }
    cudaFree(dsent);
    free(sent);
    return 0;
}
