#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp deviceProp;
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA capable devices found.\n");
        return 1;
    }

    for (int device = 0; device < deviceCount; device++) {
        cudaGetDeviceProperties(&deviceProp, device);

        printf("Device %d: %s\n", device, deviceProp.name);
        printf("-------------------------------------------------\n");
        printf("Total global memory:          %zu bytes\n", deviceProp.totalGlobalMem);
        printf("Total shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total registers per block:     %d\n", deviceProp.regsPerBlock);
        printf("Warp size:                     %d\n", deviceProp.warpSize);
        printf("Max threads per block:         %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max thread dimensions:         (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max grid dimensions:           (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Clock rate:                    %d kHz\n", deviceProp.clockRate);
        printf("Total constant memory:         %zu bytes\n", deviceProp.totalConstMem);
        printf("Texture alignment:             %zu bytes\n", deviceProp.textureAlignment);
        printf("Number of multiprocessors:     %d\n", deviceProp.multiProcessorCount);
        printf("Kernel execution timeout:      %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("-------------------------------------------------\n\n");
    }

    return 0;
}
