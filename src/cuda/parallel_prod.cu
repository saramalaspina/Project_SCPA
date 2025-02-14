#include "../../lib/utils.cu"
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>


__global__ void spmv_csr_kernel(int M, int *IRP, int *JA, double *AS, double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double sum = 0.0;
        for (int j = IRP[row]; j < IRP[row + 1]; j++) {
            sum += AS[j] * x[JA[j]];
        }
        y[row] = sum;
    }
}

double *spmv_csr_cuda(int M, int *IRP, int *JA, double *AS, double *x) {
    double *y = (double *)malloc(M * sizeof(double));
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;
    
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, IRP[M] * sizeof(int));
    cudaMalloc(&d_AS, IRP[M] * sizeof(double));
    cudaMalloc(&d_x, M * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));
    
    cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, IRP[M] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, AS, IRP[M] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, M * sizeof(double), cudaMemcpyHostToDevice);
    
    int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    spmv_csr_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);
    
    cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return y;
}

__global__ void spmv_hll_kernel(int total_rows, int max_nz, double *AS, int *JA, double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < total_rows) {
        double sum = 0.0;
        for (int j = 0; j < max_nz; j++) {
            sum += AS[row * max_nz + j] * x[JA[row * max_nz + j]];
        }
        y[row] = sum;
    }
}

double *spmv_hll_cuda(int total_rows, int max_nz, double *AS, int *JA, double *x) {
    double *y = (double *)malloc(total_rows * sizeof(double));
    double *d_AS, *d_x, *d_y;
    int *d_JA;
    
    cudaMalloc(&d_AS, total_rows * max_nz * sizeof(double));
    cudaMalloc(&d_JA, total_rows * max_nz * sizeof(int));
    cudaMalloc(&d_x, total_rows * sizeof(double));
    cudaMalloc(&d_y, total_rows * sizeof(double));
    
    cudaMemcpy(d_AS, AS, total_rows * max_nz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, total_rows * max_nz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, total_rows * sizeof(double), cudaMemcpyHostToDevice);
    
    int blocks = (total_rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    spmv_hll_kernel<<<blocks, THREADS_PER_BLOCK>>>(total_rows, max_nz, d_AS, d_JA, d_x, d_y);
    
    cudaMemcpy(y, d_y, total_rows * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_AS);
    cudaFree(d_JA);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return y;
}
