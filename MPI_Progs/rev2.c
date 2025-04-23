#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define MCW MPI_COMM_WORLD

void hErr(int err) {
    if(err != MPI_SUCCESS) {
        int len = 50, ecode; char estr[len];
        MPI_Error_class(err, &ecode);
        MPI_Error_string(err, estr, &len);
        printf("\nError Code:\t%d\nError: %s\n", ecode, estr);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    int rank, size, err;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    int x;
    if(rank == 0) {
        printf("\nEnter a number:\n");
        fflush(stdout);
        scanf("%d", &x);
        x++;
        MPI_Send(&x, 1, MPI_INT, (rank+1)%size, 0, MCW);
        MPI_Recv(&x, 1, MPI_INT, (rank-1+size)%size, 0, MCW, &stat);
        printf("\nRank %d Received: %d", rank, x);
        fflush(stdout);
    }
    else {
        MPI_Recv(&x, 1, MPI_INT, (rank-1+size)%size, 0, MCW, &stat);
        printf("\nRank %d Received: %d", rank, x++);
        fflush(stdout);
        MPI_Send(&x, 1, MPI_INT, (rank+1)%size, 0, MCW);
    }
    MPI_Finalize();
}
