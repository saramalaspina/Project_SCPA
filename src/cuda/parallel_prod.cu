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

void prodCudaCSR(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time) {
    int *IRP = csr->IRP;
    int *JA = csr->JA;
    double *AS = csr->AS;

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

    // Configurazione per il calcolo del tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    spmv_csr_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
 
    cudaEventElapsedTime(elapsed_time, start, stop);
    
    cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

}

//CSR con Warp 

/* __global__ void spmv_csr_warp_kernel(int M, int *IRP, int *JA, double *AS, double *x, double *y) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (row < M) {
        double sum = 0.0;
        int row_start = IRP[row];
        int row_end = IRP[row + 1];
        
        for (int j = row_start + lane; j < row_end; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
        }
        
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        
        if (lane == 0) y[row] = sum;
    }     
} */

__global__ void spmv_csr_warp_kernel(int num_rows, int *d_row_ptr, int *d_col_indices, double *d_values, double *d_x, double *d_y) {
     __shared__ double sdata[THREADS_PER_BLOCK];  
    __shared__ int ptrs[THREADS_PER_BLOCK / WARP_SIZE][2];

    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;  // ID globale del thread
    const int thread_in_warp = threadIdx.x & (WARP_SIZE - 1);     // ID del thread dentro il warp
    const int warp_id = thread_id / WARP_SIZE;                    // ID globale del warp
    const int warp_in_block = threadIdx.x / WARP_SIZE;            // Warp ID locale nel blocco
    const int num_warps = (THREADS_PER_BLOCK / WARP_SIZE) * gridDim.x;   // Numero totale di warps

    // Ogni warp si occupa di una riga della matrice
    for (int row = warp_id; row < num_rows; row += num_warps) {
        // I primi due thread di ogni warp caricano gli indici di inizio/fine riga
        if (thread_in_warp < 2) {
            ptrs[warp_in_block][thread_in_warp] = d_row_ptr[row + thread_in_warp];
        }
        __syncthreads(); // Sincronizzazione per garantire che ptrs sia stato aggiornato

        int row_start = ptrs[warp_in_block][0];  
        int row_end = ptrs[warp_in_block][1];    

        // Calcolo del prodotto locale
        double sum = 0.0;
        for (int j = row_start + thread_in_warp; j < row_end; j += WARP_SIZE) {
            sum += d_values[j] * d_x[d_col_indices[j]];
        }

        // Riduzione all'interno del warp
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Il primo thread del warp scrive il risultato finale nella memoria globale
        if (thread_in_warp == 0) {
            d_y[row] = sum;
        }
    }
}

void prodCudaCSRWarp(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time) {
    int *IRP = csr->IRP;
    int *JA = csr->JA;
    double *AS = csr->AS;

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

    int nz = csr->IRP[M];  // numero totale di non zeri
    double avg_nz_row = (double)nz / M; // numero medio di non zeri per riga

    // Configurazione per il calcolo del tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (avg_nz_row < 16) {
        // Lancia il kernel classico thread-per-row
        int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cudaEventRecord(start, 0);

        spmv_csr_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(elapsed_time, start, stop);

    } else {
        // Lancia il kernel warp-level: un warp per riga
        int blocks = (M * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cudaEventRecord(start, 0);

        spmv_csr_warp_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(elapsed_time, start, stop);
    } 
 
    cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_IRP);
    cudaFree(d_JA);
    cudaFree(d_AS);
    cudaFree(d_x);
    cudaFree(d_y);

}

//HLL

/* Kernel CUDA per il prodotto matrice-vettore.
   Ogni thread elabora una riga globale: calcola a quale blocco appartiene e l'indice locale,
   quindi accumula il prodotto per tutti i non-zeri in quella riga. */
__global__ void spmv_hll_kernel(int hackSize, int totalRows, EllpackBlock *d_blocks, const double *d_x, double *d_y) {
    int globalRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalRow >= totalRows) return;

    // Determina il blocco e la riga locale in base a hackSize
    int b = globalRow / hackSize;
    int localRow = globalRow % hackSize;
    if (localRow >= d_blocks[b].block_rows)
        return; // nel caso dell'ultimo blocco che contiene meno righe

    double sum = 0.0;
    int maxnz = d_blocks[b].maxnz;
    int rowStart = localRow * maxnz;
    for (int j = 0; j < maxnz; j++) {
        int col = d_blocks[b].JA[rowStart + j];
        if (col != -1) {  // -1 indica una cella vuota
            sum += d_blocks[b].AS[rowStart + j] * d_x[col];
        }
    }
    d_y[globalRow] = sum;
}


void prodCudaHLL(const HLLMatrix *hllHost, const double *xHost, double *yHost, int totalRows, float *elapsed_time) {
    double *d_x, *d_y;
    cudaMalloc((void**)&d_x, totalRows * sizeof(double));
    cudaMalloc((void**)&d_y, totalRows * sizeof(double));
    cudaMemcpy(d_x, xHost, totalRows * sizeof(double), cudaMemcpyHostToDevice);

    // Allocazione dell'array dei blocchi su device
    EllpackBlock *d_blocks;
    cudaMalloc((void**)&d_blocks, hllHost->numBlocks * sizeof(EllpackBlock));

    // Preparo una copia host (temporanea) dei blocchi con i puntatori device
    EllpackBlock *h_blocksDevice = (EllpackBlock *) malloc(hllHost->numBlocks * sizeof(EllpackBlock));
    for (int b = 0; b < hllHost->numBlocks; b++) {
        int sizeBlock = hllHost->blocks[b].block_rows * hllHost->blocks[b].maxnz;
        int *d_JA;
        double *d_AS;
        cudaMalloc((void**)&d_JA, sizeBlock * sizeof(int));
        cudaMalloc((void**)&d_AS, sizeBlock * sizeof(double));
        // Copia dei dati degli array JA e AS per il blocco corrente
        cudaMemcpy(d_JA, hllHost->blocks[b].JA, sizeBlock * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_AS, hllHost->blocks[b].AS, sizeBlock * sizeof(double), cudaMemcpyHostToDevice);

        // Imposto il blocco nel vettore temporaneo, con i puntatori aggiornati (device)
        h_blocksDevice[b].block_rows = hllHost->blocks[b].block_rows;
        h_blocksDevice[b].N = hllHost->blocks[b].N;
        h_blocksDevice[b].maxnz = hllHost->blocks[b].maxnz;
        h_blocksDevice[b].JA = d_JA;
        h_blocksDevice[b].AS = d_AS;
    }
    // Copia dell'array dei blocchi (con i puntatori device) su device
    cudaMemcpy(d_blocks, h_blocksDevice, hllHost->numBlocks * sizeof(EllpackBlock), cudaMemcpyHostToDevice);

    // Costruisco la struttura HLLMatrix sul device
    HLLMatrix hllDevice;
    hllDevice.hackSize = hllHost->hackSize;
    hllDevice.numBlocks = hllHost->numBlocks;
    hllDevice.blocks = d_blocks;
    HLLMatrix *d_hll;
    cudaMalloc((void**)&d_hll, sizeof(HLLMatrix));
    cudaMemcpy(d_hll, &hllDevice, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    // Lancio del kernel: un thread per riga globale
    int gridSize = (totalRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Configurazione per il calcolo del tempo di esecuzione
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);

    spmv_hll_kernel<<<gridSize, THREADS_PER_BLOCK>>>(hllHost->hackSize, totalRows, d_blocks, d_x, d_y);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
 
    cudaEventElapsedTime(elapsed_time, start, stop);

    // Copia del vettore risultato y da device a host
    cudaMemcpy(yHost, d_y, totalRows * sizeof(double), cudaMemcpyDeviceToHost);

    // Liberazione della memoria device per ciascun blocco
    for (int b = 0; b < hllHost->numBlocks; b++) {
        cudaFree(h_blocksDevice[b].JA);
        cudaFree(h_blocksDevice[b].AS);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_blocksDevice);
    cudaFree(d_blocks);
    cudaFree(d_hll);
    cudaFree(d_x);
    cudaFree(d_y);
}




