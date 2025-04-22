#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define MAX_VAL 25
#define MIN_VAL 1

__global__ void csrm(int* data, int* col_idx, int* row_ptr, int* vec, int* res) {
    int idx = threadIdx.x;
    int val = 0;
    for(int i = row_ptr[idx]; i < row_ptr[idx+1]; i++) {
        val += data[i] * vec[col_idx[i]];
    }
    res[idx] = val;
}

int main() {
    int m, n;
    printf("\nEnter the dimensions of OG matrix:\n");
    scanf("%d%d", &m, &n);
    int mat[m][n];
    printf("\nElements:\n");
    int num_el = 0;
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if(rand()%5 < 2) {
                mat[i][j] = 0;
            }
            else {
                num_el++;
                mat[i][j] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL; 
            }
            printf("%d ", mat[i][j]);
        }
        printf("\n");
    }

    int *data, *col_idx, *row_ptr, *vec, *res, *ddata, *dcol_idx, *drow_ptr, *dvec, *dres;
    int data_size = num_el * sizeof(int);
    int row_size = (m+1) * sizeof(int);
    int vec_size = n * sizeof(int);
    int res_size = m * sizeof(int);
    data = (int*) malloc(data_size);
    col_idx = (int*) malloc(data_size);
    row_ptr = (int*) malloc(row_size);
    vec = (int*) malloc(vec_size);
    res = (int*) malloc(res_size);
    num_el = 0;
    for(int i = 0; i < m; i++) {
        row_ptr[i] = num_el;
        for(int j = 0; j < n; j++) {
            if(mat[i][j] != 0) {
                data[num_el] = mat[i][j];
                col_idx[num_el++] = j;
            }
        }
    }
    row_ptr[m] = num_el;
    for(int i = 0; i < n; i++) {
        vec[i] = (rand() % (MAX_VAL - MIN_VAL)) + MIN_VAL;
    }
    
    printf("\nCSR Representation:\n");
    printf("\nData: ");
    for(int i = 0; i < num_el; i++) {
        printf("%d ", data[i]);
    }
    printf("\nCol Index: ");
    for(int i = 0; i < num_el; i++) {
        printf("%d ", col_idx[i]);
    }
    printf("\nRow Pointer: ");
    for(int i = 0; i <= m; i++) {
        printf("%d ", row_ptr[i]);
    }
    printf("\n\nVector: ");
    for(int i = 0; i < n; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n\n");

    cudaMalloc((void**) &ddata, data_size);
    cudaMalloc((void**) &dcol_idx, data_size);
    cudaMalloc((void**) &drow_ptr, row_size);
    cudaMalloc((void**) &dvec, vec_size);
    cudaMalloc((void**) &dres, res_size);
    cudaMemcpy(ddata, data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcol_idx, col_idx, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(drow_ptr, row_ptr, row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dvec, vec, vec_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaError_t err;
    float el_time;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    csrm<<<1,m>>> (ddata, dcol_idx, drow_ptr, dvec, dres);
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("\nError: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&el_time, start, stop);
    printf("\nElapsed time: %.3f ms", el_time);
    cudaMemcpy(res, dres, res_size, cudaMemcpyDeviceToHost);
    printf("\n\nResultant Vector:\n");
    for(int i = 0; i < m; i++) {
        printf("%d ", res[i]);
    }

    free(data); free(col_idx); free(row_ptr); free(vec); free(res);
    cudaFree(ddata); cudaFree(dcol_idx); cudaFree(drow_ptr); cudaFree(dvec); cudaFree(dres);
}