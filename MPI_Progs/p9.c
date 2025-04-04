// 4x4 parallel adder

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define MCW MPI_COMM_WORLD

void callErr(int err) {
    if(err != 0) {
        char estr[50]; int len = 50;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s", estr);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    int rank, size, err;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    int mat[size][size], temp[size], inter[size];
    if(rank == 0) {
        printf("Enter the %d x %d matrix:\n", size, size);
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                scanf("%d", &mat[i][j]);
            }
        }
        printf("\nCurrent Matrix:\n");
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                printf("%d ", mat[i][j]);
            }
            printf("\n");
        }
        fflush(stdout);
    }
    err = MPI_Scatter(mat, size, MPI_INT, temp, size, MPI_INT, 0, MCW);
    callErr(err);
    err = MPI_Scan(temp, inter, size, MPI_INT, MPI_SUM, MCW);
    callErr(err);
    printf("\nRank %d: ",rank);
    for(int i = 0; i < size; i++) {
        printf("%d ", inter[i]);
    }
    printf("\n");
    fflush(stdout);
    MPI_Finalize();
}