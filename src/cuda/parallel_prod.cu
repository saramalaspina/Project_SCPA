#include "../../lib/utils.h"
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

// ================= CSR =================

// CUDA kernel for matrix-vector product in CSR format: one thread per row
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

// CUDA kernel for matrix-vector product in CSR format with warp-level parallelism: one warp per row
__global__ void spmv_csr_warp_kernel(int M, int *IRP, int *JA, double *AS, double *x, double *y) {
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
}

// Host function to compute the product using CUDA with CSR format
void prod_cuda_csr(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time) {
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

    int nz = csr->IRP[M];
    // Average number of non-zeros per row
    double avg_nz_row = (double)nz / M; 

    // Configure timers to measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (avg_nz_row < 16) {
        // Launch the thread-per-row kernel
        int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cudaEventRecord(start, 0);

        spmv_csr_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
           fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
           exit(EXIT_FAILURE);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(elapsed_time, start, stop);

    } else {
        // Launch warp-level kernel
        int blocks = (M * WARP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaEventRecord(start, 0);

        spmv_csr_warp_kernel<<<blocks, THREADS_PER_BLOCK>>>(M, d_IRP, d_JA, d_AS, d_x, d_y);
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error (warp-level): %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

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

// ================= HLL =================

/* CUDA kernel for matrix-vector product in HLL format: each thread processes a global row */
__global__ void spmv_hll_kernel(int hackSize, int totalRows, EllpackBlock *d_blocks, const double *d_x, double *d_y) {
    int globalRow = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalRow >= totalRows) return;

    // Determine which block the global row belongs to and the local row index within the block
    int b = globalRow / hackSize;
    int localRow = globalRow % hackSize;

    if (localRow >= d_blocks[b].block_rows)
        return;

    double sum = 0.0;
    int maxnz = d_blocks[b].maxnz;
    int blockRows = d_blocks[b].block_rows;

    // Iterate over all the entries in the row of the block b
    for (int j = 0; j < maxnz; j++) {
        // Accessing in column-major format
        int index = j * blockRows + localRow;
        // Read the column index
        int col = d_blocks[b].JA[index];
        // Check if the index is valid (not a padded entry)
        if (col != -1) {
            sum += d_blocks[b].AS[index] * d_x[col];
        }
    }
    d_y[globalRow] = sum;
}

// Optimized CUDA kernel with warp-level parallelism
__global__ void spmv_hll_kernel_warp(int hackSize, int totalRows, EllpackBlock *d_blocks, const double *d_x, double *d_y) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp processes one row
    int warpId = threadId / warpSize;  
    // Lane index within the warp    
    int lane   = threadIdx.x % warpSize;    

    if (warpId >= totalRows) return;

    int b = warpId / hackSize;
    int localRow = warpId % hackSize;

    if (localRow >= d_blocks[b].block_rows)
        return;

    int maxnz = d_blocks[b].maxnz;
    int blockRows = d_blocks[b].block_rows;

    double sum = 0.0;

    // Each lane processes part of the columns
    for (int j = lane; j < maxnz; j += warpSize) {
        int index = j * blockRows + localRow; 
        int col = d_blocks[b].JA[index];
        if (col != -1) {
            sum += d_blocks[b].AS[index] * d_x[col];
        }
    }

    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane == 0) {
        d_y[warpId] = sum;
    }
}

// Host function to compute matrix-vector product using CUDA and HLL format
void prod_cuda_hll(const HLLMatrix *hllHost, const double *xHost, double *yHost, int totalRows, float *elapsed_time) {
    int N = hllHost->blocks[0].N;

    // Allocate and copy vectors x and y on device
    double *d_x, *d_y;
    cudaMalloc((void**)&d_x, totalRows * sizeof(double));
    cudaMalloc((void**)&d_y, totalRows * sizeof(double));
    cudaMemcpy(d_x, xHost, N * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate block array on device
    EllpackBlock *d_blocks;
    cudaMalloc((void**)&d_blocks, hllHost->numBlocks * sizeof(EllpackBlock));

    // Create a temporary host copy of blocks with device pointers
    EllpackBlock *h_blocksDevice = (EllpackBlock *) malloc(hllHost->numBlocks * sizeof(EllpackBlock));
    for (int b = 0; b < hllHost->numBlocks; b++) {
        int sizeBlock = hllHost->blocks[b].block_rows * hllHost->blocks[b].maxnz;
        int *d_JA;
        double *d_AS;
        cudaMalloc((void**)&d_JA, sizeBlock * sizeof(int));
        cudaMalloc((void**)&d_AS, sizeBlock * sizeof(double));

        // Copy JA and AS arrays for the current block
        cudaMemcpy(d_JA, hllHost->blocks[b].JA, sizeBlock * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_AS, hllHost->blocks[b].AS, sizeBlock * sizeof(double), cudaMemcpyHostToDevice);

        // Set the block in the temp array with updated device pointers
        h_blocksDevice[b].block_rows = hllHost->blocks[b].block_rows;
        h_blocksDevice[b].N = hllHost->blocks[b].N;
        h_blocksDevice[b].maxnz = hllHost->blocks[b].maxnz;
        h_blocksDevice[b].JA = d_JA;
        h_blocksDevice[b].AS = d_AS;
    }

    // Copy the block array to the device
    cudaMemcpy(d_blocks, h_blocksDevice, hllHost->numBlocks * sizeof(EllpackBlock), cudaMemcpyHostToDevice);

    // Construct the HLLMatrix structure on device
    HLLMatrix hllDevice;
    hllDevice.hackSize = hllHost->hackSize;
    hllDevice.numBlocks = hllHost->numBlocks;
    hllDevice.blocks = d_blocks;
    HLLMatrix *d_hll;
    cudaMalloc((void**)&d_hll, sizeof(HLLMatrix));
    cudaMemcpy(d_hll, &hllDevice, sizeof(HLLMatrix), cudaMemcpyHostToDevice);

    // Configure timers for performance measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel: one thread per global row
    int gridSize = (totalRows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start, 0);

    spmv_hll_kernel<<<gridSize, THREADS_PER_BLOCK>>>(hllHost->hackSize, totalRows, d_blocks, d_x, d_y);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(elapsed_time, start, stop);

    // Copy result vector y from device to host
    cudaMemcpy(yHost, d_y, totalRows * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory for each block
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