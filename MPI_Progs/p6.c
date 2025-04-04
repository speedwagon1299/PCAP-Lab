// split array into processes and count 
// number of even(1) and odd(1) in root

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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size, err, len;
    MPI_Comm_set_errhandler(MCW, MPI_ERRORS_RETURN);  // MPI_Errhandler_set
    MPI_Comm_rank(MCW, &rank);
    MPI_Comm_size(MCW, &size);
    MPI_Status stat;
    int arr[50], temp[10], m;
    if(rank == 0) {
        printf("Enter the number of elements:\n");
        scanf("%d", &m);
        for(int i = 0; i < m; i++) {
            scanf("%d", &arr[i]);
        }
        printf("\nInputted Array:\n");
        for(int i = 0; i < m; i++) {
            printf("%d, ", arr[i]);
        }
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MCW);
    MPI_Scatter(arr, m/size, MPI_INT, temp, m/size, MPI_INT, 0, MCW);
    for(int i = 0; i < m/size; i++) {
        temp[i] = (temp[i]%2 == 0);
    }
    MPI_Gather(temp, m/size, MPI_INT, arr, m/size, MPI_INT, 0, MCW);
    if(rank == 0) {
        printf("\nResultant Array:\n");
        int even = 0, odd = 0;
        for(int i = 0; i < m; i++) {
            printf("%d, ", arr[i]);
            if(arr[i] == 1) {
                even++;
            }
            else {
                odd++;
            }
        }
        printf("\nEven:\t%d\nOdd:\t%d\n", even, odd);
    }
    MPI_Finalize();
}