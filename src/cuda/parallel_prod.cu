#include "../../lib/utils.h"
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

double *spmv_csr_cuda(int M, int N, int *IRP, int *JA, double *AS, double *x) {
    double *y = (double *)malloc(M * sizeof(double));
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;
    
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, IRP[M] * sizeof(int));
    cudaMalloc(&d_AS, IRP[M] * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));
    
    cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, IRP[M] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, AS, IRP[M] * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    spmv_csr_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize(); 
    
    cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);
    
    return y;
}

// Kernel CUDA per il prodotto matrice-vettore in formato HLL
__global__ void spmv_hll_kernel(ELLBlockDevice *d_blocks, int num_blocks, const double *d_x, double *d_y) {
    int b = blockIdx.x;  // Ogni CUDA block elabora un blocco HLL
    if (b < num_blocks) {
        ELLBlockDevice block = d_blocks[b];
        int rows   = block.rows;
        int max_nz = block.max_nz;
        
        int local_row = threadIdx.x;  // Ogni thread elabora una riga del blocco
        if (local_row < rows) {
            double sum = 0.0;
            // Layout trasposto: per la riga "local_row", gli elementi sono in posizioni: j * rows + local_row
            for (int j = 0; j < max_nz; j++) {
                int index = j * rows + local_row;
                int col   = block.JA_flat[index];
                double val = block.AS_flat[index];
                sum += val * d_x[col];
            }
            // Calcolo dell'indice globale della riga
            int global_row = b * HACKSIZE + local_row;
            d_y[global_row] = sum;
        }
    }
}

void spmv_hll_cuda(HLLMatrixDevice *d_hll, const double *d_x, double *d_y) {
    int numBlocks = d_hll->num_blocks;
    dim3 grid(numBlocks);
    dim3 block(HACKSIZE);  // Si lancia HACKSIZE thread per blocco
    
    spmv_hll_kernel<<<grid, block>>>(d_hll->blocks, numBlocks, d_x, d_y);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Errore di lancio kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
}
