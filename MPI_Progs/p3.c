// Input array of strings at process 0 and distribute to processes
// At each process do String reversal + change text case
// Then String concatenation one by one through point to point communication from process 1, 2 ... cycling back to 0
// Display final result at process 0

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#define MCW MPI_COMM_WORLD
#define ELEN 50

void callErr(int err) {
    if(err != 0) {
        char estr[ELEN]; int len = ELEN;
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s", estr);
    }
}

void swap(char* a, char* b) {
    char temp = *a;
    *a = *b;
    *b = temp;
}

void toggle(char* word) {
    int len = strlen(word);
    for(int i = 0; i < len; i++) {
        if(word[i] >= 'a' && word[i] <= 'z') {
            word[i] -= 32;
        }
        else {
            word[i] += 32;
        }
    }
}

void reverse(char* word) {
    int len = strlen(word);
    for(int i = 0; i < len/2; i++) {
        swap(&word[i], &word[len-i-1]);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, err, bufsiz, len;
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);  // MPI_Errhandler_set
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    char inp[size][50];
    char buf[50], *buf2;
    char final[size*50];
    if(rank == 0) {
        for(int i = 0; i < size; i++) {
            scanf("%s ", inp[i]);
        }
    }
    MPI_Scatter(*inp, 50, MPI_CHAR, buf, 50, MPI_CHAR, 0, MCW);
    reverse(buf);
    toggle(buf);
    printf("Rank %d: %s", rank, buf);
    if(rank == 0) {
        strcpy(final, buf);
        len = strlen(final);
        bufsiz = len*sizeof(char) + MPI_BSEND_OVERHEAD;
        buf2 = (char*) malloc(bufsiz);
        MPI_Ssend(&len, 1, MPI_INT, 1, 0, MCW);
        MPI_Buffer_attach(buf2, bufsiz);
        MPI_Bsend(final, len + 1, MPI_CHAR, 1, 0, MCW);
        MPI_Buffer_detach(&buf2, &bufsiz);
        MPI_Recv(&len, 1, MPI_INT, size-1, 0, MCW, &stat);
        MPI_Recv(final, len+1, MPI_CHAR, size-1, 0, MCW, &stat);
        printf("\nFinal Result: %s\n", final);
    }
    else {
        MPI_Recv(&len, 1, MPI_INT, rank-1, 0, MCW, &stat);
        MPI_Recv(final, len + 1, MPI_CHAR, rank-1, 0, MCW, &stat);
        strcat(final, buf);
        len = strlen(final);
        bufsiz = len*sizeof(char) + MPI_BSEND_OVERHEAD;
        buf2 = (char*) malloc(bufsiz);
        MPI_Ssend(&len, 1, MPI_INT, (rank+1)%size, 0, MCW);
        MPI_Buffer_attach(buf2, bufsiz);
        MPI_Bsend(final, len + 1, MPI_CHAR, (rank+1)%size, 0, MCW);
        MPI_Buffer_detach(&buf2, &bufsiz);
    }
    MPI_Finalize();
}