#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
    // think in parallel. threadIdx is parallel, as is blockIdx
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a[tid] + b[tid];
    }
    

}

int main(){
    float *a, *b, *out; 
    float *a_gp, *b_gp, *out_gp;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&a_gp, sizeof(float) * N);
    cudaMalloc((void**)&b_gp, sizeof(float) * N);
    cudaMalloc((void**)&out_gp, sizeof(float) * N);


    // Transfer data
    //cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)
    cudaMemcpy(a_gp, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gp, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Kernel function
    int sm=1; //i have 40. But I should do N/
    int block_size = 25; //block size or grid_size
    int grid_size = ((N + block_size) / block_size);

    vector_add<<<grid_size,block_size>>>(out_gp, a_gp, b_gp, N); 

    // Transfer data back
    cudaMemcpy(out, out_gp, sizeof(float)*N, cudaMemcpyDeviceToHost);

    // Verify
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

    // free memory
    cudaFree(a_gp);
    cudaFree(b_gp);
    cudaFree(out_gp);

    free(a);
    free(b);
    free(out);
}