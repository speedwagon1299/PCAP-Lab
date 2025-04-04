// 1! + 2! + ...


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#define MCW MPI_COMM_WORLD
#define ELEN 50

int fact(int num) {
    if(num <= 1) {
        return 1;
    }
    return num * fact(num-1);
}

void callErr(int err) {
    if(err != 0) {
        char estr[ELEN]; int len = ELEN;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s", estr);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, err;
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);  // MPI_Errhandler_set
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    int num, ans;
    MPI_Status stat;
    num = fact(rank+1);
    err = MPI_Scan(&num, &ans, 1, MPI_INT, MPI_SUM, MCW);
    callErr(err);
    printf("\nRank %d: %d", rank, ans);
    MPI_Finalize();
}