#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#define MCW MPI_COMM_WORLD

void callErr(int err) {
    if(err != 0) {
        char estr[50]; int len = 50;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s\n", estr);
    }
}

int func(int* temp, int op, int m) {
    int res = temp[0], len = m;
    switch(op) {
        case 1:
            for(int i = 1; i < len; i++) {
                res += temp[i];
            }
            break;
        case 2:
            for(int i = 1; i < len; i++) {
                res *= temp[i];
            }
            break;
    }
    return res;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, err;
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    int bufsiz, m, op[size], res, ans;
    int *buf;
    if(rank == 0) {
        printf("\nEnter the rowsize:\n");
        fflush(stdout);
        scanf("%d", &m);
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MCW);
    int mat[size][m], temp[m];
    if(rank == 0) {
        printf("\nEnter the %d x %d elements:\n", size, m);
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < m; j++) {
                scanf("%d", &mat[i][j]);
            }
        }
        printf("\nEnter the operation for each row:\t1 (Add), 2 (Multiply)\n");
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            scanf("%d", &op[i]);
        }
        bufsiz = sizeof(int) + MPI_BSEND_OVERHEAD;
        buf = (int*) malloc(bufsiz);
        err = MPI_Buffer_attach(buf, bufsiz);
        callErr(err);
        for(int i = 1; i < size; i++) {
            err = MPI_Bsend(&op[i], 1, MPI_INT, i, 0, MCW);
            callErr(err);
        }
        err = MPI_Buffer_detach(&buf, &bufsiz);
        callErr(err);
    }
    else {
        err = MPI_Recv(&op[rank], 1, MPI_INT, 0, 0, MCW, &stat);
        callErr(err);
    }
    err = MPI_Scatter(mat, m, MPI_INT, temp, m, MPI_INT, 0, MCW);
    callErr(err);
    res = func(temp, op[rank], m);
    err = MPI_Scan(&res, &ans, 1, MPI_INT, MPI_SUM, MCW);
    printf("\nRank %d:\t%d", rank, ans);
    if(rank == size-1) {
        printf("\nFinal Answer: %d", ans);
    }
    MPI_Finalize();
}