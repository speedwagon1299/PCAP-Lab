// read M and NxM matrix. Square of first M, Cube of second M, ...

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MCW MPI_COMM_WORLD
#define ELEN 50

void callErr(int err) {
    if(err != 0) {
        int len = 50; char estr[50];
        MPI_Error_string(err, estr, &len);
        printf("Error %d:\t%s", err, estr);
    }
}

int main(int argc, char* argv[]) {
    int rank, n, err;
    MPI_Init(&argc,&argv);
    MPI_Errhandler_set(MCW, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &n);
    MPI_Status stat;
    int* mat, *temp, m;
    if(rank == 0) {
        printf("Enter the value of M:\n");
        scanf("%d",&m);
        mat = (int*) calloc(m*n, sizeof(int));
    }
    err = MPI_Bcast(&m, 1, MPI_INT, 0, MCW);
    callErr(err);
    temp = (int*) calloc(m, sizeof(int));
    if(rank == 0) {
        printf("\nEnter the elements:\n");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                scanf("%d", &mat[i*m + j]);
            }
        }
    }
    err = MPI_Scatter(mat, m, MPI_INT, temp, m, MPI_INT, 0, MCW);
    callErr(err);
    for(int i = 0; i < m; i++) {
        temp[i] = (int) pow(temp[i], rank+2);
    }
    err = MPI_Gather(temp, m, MPI_INT, mat, m, MPI_INT, 0, MCW);
    callErr(err);
    if(rank == 0) {
        printf("\nResultant Matrix:\n");
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                printf("%d ", mat[i*m + j]);
            }
            printf("\n");
        }
    }
    MPI_Finalize();
    exit(0);
}