#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void rowMul(int* a, int* b, int* c, int ra, int ca, int cb) {
    int rid = threadIdx.x;
    int val;
    for(int i = 0; i < cb; i++) {
        val = 0;
        for(int j = 0; j < ca; j++) {
            val += a[rid*ca + j] * b[j*cb + i];
        }
        c[rid*cb + i] = val;
    }
}

__global__ void colMul(int* a, int* b, int* c, int ra, int ca, int cb) {
    int cid = threadIdx.y;
    int val;
    for(int i = 0; i < ra; i++) {
        val = 0;
        for(int j = 0; j < ca; j++) {
            val += a[i*ca + j] * b[j*cb + cid];
        }
        c[i*cb + cid] = val;
    }
}

__global__ void elMul(int* a, int* b, int* c, int ra, int ca, int cb) {
    int rid = threadIdx.x, cid = threadIdx.y;
    int val = 0;
    for(int j = 0; j < ca; j++) {
        val += a[rid*ca + j] * b[j*cb + cid];
    }
    c[rid*cb + cid] = val;
}

__host__ void printMat(int* a, int r, int c) {
    printf("\n");
    for(int i = 0; i < r; i++) {
        for(int j = 0; j < c; j++) {
            printf("%d ", a[i*c + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int *a, *b, *c, *da, *db, *dc, ra, ca, rb, cb;
    printf("\nEnter dimensions of A and B respectively:\n");
    scanf("%d%d%d%d", &ra, &ca, &rb, &cb);
    ca = rb;
    int a_size = ra * ca * sizeof(int);
    int b_size = rb * cb * sizeof(int);
    int c_size = ra * cb * sizeof(int);
    a = (int*) malloc(a_size);
    b = (int*) malloc(b_size);
    c = (int*) malloc(c_size);
    for(int i = 0; i < ra; i++) {
        for(int j = 0; j < ca; j++) {
            a[i*ca + j] = rand() % 15;
        }
    }
    for(int i = 0; i < rb; i++) {
        for(int j = 0; j < cb; j++) {
            b[i*cb + j] = rand() % 15;
        }
    }

    printf("\nA:\n");
    printMat(a, ra, ca);
    printf("\nB:\n");
    printMat(b, rb, cb);

    cudaMalloc((void**) &da, a_size);
    cudaMalloc((void**) &db, b_size);
    cudaMalloc((void**) &dc, c_size);
    cudaMemcpy(da, a, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, b_size, cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaError_t err;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float el_time;
    dim3 rowDim(ra,1,1);
    dim3 colDim(1,cb,1);
    dim3 elDim(ra,cb,1);

    cudaEventRecord(start, 0);
    rowMul<<<1, rowDim>>> (da, db, dc, ra, ca, cb);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\nError in Row-Mul: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(c, dc, c_size, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&el_time, start, stop);
    printf("\n\nRow-wise:\n");
    printMat(c, ra, cb);
    printf("Elapsed time: %.3f ms", el_time);

    cudaEventRecord(start, 0);
    colMul<<<1, colDim>>> (da, db, dc, ra, ca, cb);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\nError in Col-Mul: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(c, dc, c_size, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&el_time, start, stop);
    printf("\n\nColumn-wise:\n");
    printMat(c, ra, cb);
    printf("Elapsed time: %.3f ms", el_time);

    cudaEventRecord(start, 0);
    elMul<<<1, elDim>>> (da, db, dc, ra, ca, cb);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\nError in El-Mul: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaMemcpy(c, dc, c_size, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&el_time, start, stop);
    printf("\n\nElement-wise:\n");
    printMat(c, ra, cb);
    printf("Elapsed time: %.3f ms", el_time);

    free(a); free(b); free(c);
    cudaFree(da); cudaFree(db); cudaFree(dc);
}