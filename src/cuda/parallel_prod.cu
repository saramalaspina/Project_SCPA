#include "../../lib/utils.h"
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

//CSR

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

double *prodCudaCSR(int M, int N, int *IRP, int *JA, double *AS, double *x) {
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

//HLL

__global__ void spmv_hll_kernel(int rows, int max_nz, const int *JA_t, const double *AS_t, const double *x, double *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        // Per ogni "colonna" (cioè ogni posizione nella riga ELLPACK)
        for (int j = 0; j < max_nz; j++) {
            // Poiché i dati sono trasposti, l’accesso è:
            // elemento (row,j) in formato ELLPACK --> JA_t[j*rows + row] e AS_t[j*rows + row]
            int col = JA_t[j * rows + row];
            double val = AS_t[j * rows + row];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

double *prodCudaHLL(const HLLMatrix *hll, int total_rows, int total_cols, const double *x) {
    // Allocazione del vettore risultato sul host
    double *y = (double*)malloc(total_rows * sizeof(double));

    // Allocazione e copia del vettore x sul device
    double *d_x, *d_y;
    cudaMalloc(&d_x, total_cols * sizeof(double));
    cudaMalloc(&d_y, total_rows * sizeof(double));
    cudaMemcpy(d_x, x, total_cols * sizeof(double), cudaMemcpyHostToDevice);

    int row_offset = 0;  // per posizionare i risultati parziali all’interno di y

    // Processa ciascun blocco HLL (ognuno contiene un blocco in formato ELLPACK)
    for (int b = 0; b < hll->num_blocks; b++) {
        // Puntatore al blocco corrente
        ELLBlock *block = &(hll->blocks[b]);
        int rows = block->rows;
        int max_nz = block->max_nz;

        // Dimensione dei dati trasposti per il blocco
        size_t size_int = rows * max_nz * sizeof(int);
        size_t size_double = rows * max_nz * sizeof(double);

        // Alloca memoria sul device per JA_t e AS_t del blocco
        int *d_JA_t;
        double *d_AS_t;
        cudaMalloc(&d_JA_t, size_int);
        cudaMalloc(&d_AS_t, size_double);

        // Copia dei dati del blocco sul device
        cudaMemcpy(d_JA_t, block->JA_t, size_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_AS_t, block->AS_t, size_double, cudaMemcpyHostToDevice);

        // Calcola la configurazione di esecuzione per il kernel
        int numBlocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        // Il kernel scrive in d_y a partire da d_y + row_offset
        spmv_hll_kernel<<<numBlocks, THREADS_PER_BLOCK>>>(rows, max_nz, d_JA_t, d_AS_t, d_x, d_y + row_offset);
        cudaDeviceSynchronize();

        // Libera la memoria allocata per il blocco
        cudaFree(d_JA_t);
        cudaFree(d_AS_t);

        row_offset += rows;
    }

    // Copia del vettore risultato dal device al host
    cudaMemcpy(y, d_y, total_rows * sizeof(double), cudaMemcpyDeviceToHost);

    // Libera la memoria sul device
    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}




