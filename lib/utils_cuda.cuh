#ifndef UTILS_CUDA_CUH
#define UTILS_CUDA_CUH

#include <cuda_runtime.h>

//parallel product cuda 

#define THREADS_PER_BLOCK 256

__global__ void spmv_csr_kernel(int M, int *IRP, int *JA, double *AS, double *x, double *y);
__global__ void spmv_hll_kernel(int total_rows, int max_nz, double *AS, int *JA, double *x, double *y);
double *spmv_csr_cuda(int M, int *IRP, int *JA, double *AS, double *x);
double *spmv_hll_cuda(int total_rows, int max_nz, double *AS, int *JA, double *x);

#endif //UTILS_CUDA_CUH