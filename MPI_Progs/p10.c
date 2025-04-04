#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#define MCW MPI_COMM_WORLD

void callErr(int err) {
    if(err != 0) {
        char estr[50]; int len = 50;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s\n", estr);
    }
}

void swap(char* a, char* b) {
    char temp = *a;
    *a = *b;
    *b = temp;
}

void function(char* buf, int num) {
    int len = strlen(buf);
    for(int i = 0; i < len/2; i++) {
        swap(&buf[i], &buf[len-i-1]);
    }
    buf[len] = num + '0';
    buf[len+1] = '\0';
    for(int i = 1; i <= len+1; i++) {
        buf[i-1] = buf[i];
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    int rank, size, err;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    char sent[size][50], buf[50];

    if(rank == 0) {
        printf("Enter the sentence:\n");
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            scanf("%s ", &sent[i]);
        }
    }
    err = MPI_Scatter(*sent, 50, MPI_CHAR, buf, 50, MPI_CHAR, 0, MCW);
    callErr(err);
    function(buf, rank);
    err = MPI_Gather(buf, 50, MPI_CHAR, *sent, 50, MPI_CHAR, 0, MCW);
    callErr(err);
    if(rank == 0) {
        printf("Output String:\n");
        fflush(stdout);
        for(int i = 0; i < size; i++) {
            printf("%s ", sent[i]);
        }
        fflush(stdout);
    }
    MPI_Finalize();
}