#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <ctype.h>
#include <math.h>
#define MCW MPI_COMM_WORLD

void hErr(int err) {
    if(err != MPI_SUCCESS) {
        int len = 50, ecode; char estr[len];
        MPI_Error_string(err, estr, &len);
        printf("\nError: %s", estr);
        MPI_Error_class(err, &ecode);
        printf("\nError Code: %d", ecode);
    }
}

int isPrime(int num) { 
    int flag = 1;
    for(int i = 2; i <= sqrt(num); i++) {
        if(num%i == 0) {
            flag = 0;
            break;
        }
    }
    return flag;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    // MPI_Errhandler_set(MCW, MPI_ERRORS_RETURN);
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);
    MPI_Status stat;
    int low, high;
    if(rank == 0) {
        printf("\nEnter the range:\n");
        fflush(stdout);
        scanf("%d %d", &low, &high);
    }
    MPI_Bcast(&low, 1, MPI_INT, 0, MCW);
    MPI_Bcast(&high, 1, MPI_INT, 0, MCW);
    int buf_size = (high - low + 1) / size;
    int arr[high - low + 1], temp[buf_size];
    if(rank == 0) {
        for(int i = low; i < high; i++) {
            arr[i - low] = i;
        }
    }
    MPI_Scatter(arr, buf_size, MPI_INT, temp, buf_size, MPI_INT, 0, MCW);
    for(int i = 0; i < buf_size; i++) {
        temp[i] = isPrime(temp[i]);
    }
    MPI_Gather(temp, buf_size, MPI_INT, arr, buf_size, MPI_INT, 0, MCW);
    if(rank == 0) {
        printf("\nPrime numbers are:\n");
        for(int i = low; i < high; i++) {
            if(arr[i-low] == 1) {
                printf("%d ", i);
            }
        }
    }
    MPI_Finalize();
}