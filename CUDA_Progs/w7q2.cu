#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string.h>

__global__ void RS(char* word, char* ans, int len) {
    int idx = blockIdx.x;
    int st = 0;
    for(int i = len; i > len-idx; i--) {
        st += i;
    }
    for(int i = st; i < st + len - idx; i++) {
        ans[i] = word[i-st];
    }
}

int main() {
    char *d_word, *word, *d_ans, *ans;
    word = (char*) malloc(50 * sizeof(char));
    printf("\nEnter the word:\n");
    scanf("%s", word);
    int n = (strlen(word) + 1) * sizeof(char);
    int anslen = ((n*(n-1)/2) + 1) * sizeof(char);
    ans = (char*) malloc(anslen);
    cudaMalloc((void**) &d_word, n);
    cudaMalloc((void**) &d_ans, anslen);
    cudaMemcpy(d_word, word, n, cudaMemcpyHostToDevice);
    RS<<<n-1,1>>>(d_word, d_ans, n-1);
    cudaMemcpy(ans, d_ans, anslen, cudaMemcpyDeviceToHost);
    ans[anslen-1] = '\0';
    printf("\nFinal Word: %s\n", ans);
    cudaFree(d_word);
    cudaFree(d_ans);
    free(word);
    free(ans);
    return 0;
}