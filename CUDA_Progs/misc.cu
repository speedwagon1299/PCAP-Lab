#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main() {
    int dev_count;
    cudaDeviceProp dev_prop;
    cudaGetDeviceCount(&dev_count);
    for(int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
        printf("maxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock);
        printf("multiProcessorCount: %d\n", dev_prop.multiProcessorCount);
        printf("totalConstMem: %ld\n", dev_prop.totalConstMem);
        printf("clockRate: %d\n", dev_prop.clockRate);
        printf("maxThreadsDim X: %d\n", dev_prop.maxThreadsDim[0]);
        printf("maxThreadsDim Y: %d\n", dev_prop.maxThreadsDim[1]);
        printf("maxThreadsDim Z: %d\n", dev_prop.maxThreadsDim[2]);
        printf("maxGridSize X: %d\n", dev_prop.maxGridSize[0]);
        printf("maxGridSize Y: %d\n", dev_prop.maxGridSize[1]);
        printf("maxGridSize Z: %d\n", dev_prop.maxGridSize[2]);
    }
}