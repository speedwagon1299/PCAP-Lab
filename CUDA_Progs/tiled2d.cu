#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define TILE_WIDTH 16
#define MAX_VAL 100
#define MIN_VAL 2

__global__ void tileMat(int* a, int* b, int* c, int ar, int ac, int bc) {
    // blockDim.x = blockDim.y = TILE_WIDTH = 16
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ int sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ int sh_B[TILE_WIDTH][TILE_WIDTH];
    int sum = 0;
    int num_tiles = (ac + TILE_WIDTH - 1) / TILE_WIDTH;
    for(int t = 0; t < num_tiles; t++) {
        // for A * B, A has constant rid for the thread
        int aRow = rid;     
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if(aRow < ar && aCol < ac) {
            sh_A[threadIdx.y][threadIdx.x] = a[aRow*ac + aCol];
        }
        else {
            sh_A[threadIdx.y][threadIdx.x] = 0;
        }
        int bRow = t * TILE_WIDTH + threadIdx.y;
        // for A * B, B has constant cid for the thread
        int bCol = cid;     
        if(bRow < ac && bCol < bc) {
            sh_B[threadIdx.y][threadIdx.x] = b[bRow * bc + bCol];
        }
        else {
            sh_B[threadIdx.y][threadIdx.x] = 0;
        }
        // since all threads have to fill the current phase's tile
        __syncthreads();
        
        for(int k = 0; k < TILE_WIDTH; k++) {
            sum += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];
        }
        // since partial sum must be completed fully
        __syncthreads();
    }
    if(rid < ar && cid < bc) {
        c[rid*bc + cid] = sum;
    }
}

int main() {
    int *a, *b, *c, *da, *db, *dc;
    int ar, ac, br, bc;
    printf("\nEnter the dimensions of A and B respectively:\n");
    scanf("%d %d %d %d", &ar, &ac, &br, &bc);
    int a_size = ar * ac * sizeof(int);
    int b_size = br * bc * sizeof(int);
    int c_size = ar * bc * sizeof(int);
    
    a = (int*) malloc(a_size);
    b = (int*) malloc(b_size);
    c = (int*) malloc(c_size);

    printf("\nA:\n");
    for(int i = 0; i < ar*ac; i++) {
        a[i] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
        printf("%d ", a[i]);
        if((i+1)%ac == 0) {
            printf("\n");
        }
    }
    printf("\nB:\n");
    for(int i = 0; i < br*bc; i++) {
        b[i] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
        printf("%d ", b[i]);
        if((i+1)%bc == 0) {
            printf("\n");
        }
    }

    cudaMalloc((void**) &da, a_size);
    cudaMalloc((void**) &db, b_size);
    cudaMalloc((void**) &dc, c_size);
    cudaMemcpy(da, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, b_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    // columns first, rows second in general
    dim3 dimGrid((bc+TILE_WIDTH-1)/TILE_WIDTH, (ar+TILE_WIDTH-1)/TILE_WIDTH);
    tileMat<<<dimGrid, dimBlock>>> (da, db, dc, ar, ac, bc);
    cudaMemcpy(c, dc, c_size, cudaMemcpyDeviceToHost);
    
    printf("\nC:\n");
    for(int i = 0; i < ar*bc; i++) {
        printf("%d ", c[i]);
        if((i+1)%bc == 0) {
            printf("\n");
        }
    }

    free(a); free(b); free(c);
    cudaFree(da); cudaFree(db); cudaFree(dc);
}
