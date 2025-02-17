#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }

    MatrixElement *mat = read_matrix(argv[1]);
    if (!mat) exit(1);

    // Creazione del mediatore
    MatrixConversionMediator mediator = createMatrixMediator();

    // Creazione struct formato CSR
    CSRMatrix csr;

    // Dichiarazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);
    packHLLMatrixForGPU(&hll);

    // Allocazione dinamica del vettore x
    double *x = (double *)malloc(mat->N * sizeof(double));
    if (!x) {
        fprintf(stderr, "Errore nell'allocazione di memoria per x.\n");
        return 1;
    }

    for (int i = 0; i < mat->N; i++) {
        x[i] = i + 1;
    }

    double *res = spmv_csr(mat->M, &csr, x);
    printf("Risultato calcolo seriale CSR:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res[i]);
    }

    // **Esecuzione su GPU (CSR)**
    double *res_csr_cuda = spmv_csr_cuda(mat->M, mat->N, csr.IRP, csr.JA, csr.AS, x);
    printf("Risultato calcolo CUDA CSR:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res_csr_cuda[i]);
    }

    int num_blocks = hll.num_blocks;
    ELLBlockDevice *h_blockDevices = (ELLBlockDevice *) malloc(num_blocks * sizeof(ELLBlockDevice));
    for (int b = 0; b < num_blocks; b++) {
        int rows = hll.blocks[b].rows;
        int max_nz = hll.blocks[b].max_nz;
        int size_flat_int = rows * max_nz * sizeof(int);
        int size_flat_double = rows * max_nz * sizeof(double);
        
        int *d_JA_flat;
        double *d_AS_flat;
        cudaMalloc((void**)&d_JA_flat, size_flat_int);
        cudaMalloc((void**)&d_AS_flat, size_flat_double);
        cudaMemcpy(d_JA_flat, hll.blocks[b].JA_flat, size_flat_int, cudaMemcpyHostToDevice);
        cudaMemcpy(d_AS_flat, hll.blocks[b].AS_flat, size_flat_double, cudaMemcpyHostToDevice);
        
        h_blockDevices[b].rows = rows;
        h_blockDevices[b].cols = hll.blocks[b].cols;
        h_blockDevices[b].max_nz = max_nz;
        h_blockDevices[b].JA_flat = d_JA_flat;
        h_blockDevices[b].AS_flat = d_AS_flat;
    }

    ELLBlockDevice *d_blocks;
    cudaMalloc((void**)&d_blocks, num_blocks * sizeof(ELLBlockDevice));
    cudaMemcpy(d_blocks, h_blockDevices, num_blocks * sizeof(ELLBlockDevice), cudaMemcpyHostToDevice);
    
    // Costruiamo la struttura HLLMatrixDevice
    HLLMatrixDevice h_hllDevice;
    h_hllDevice.num_blocks = num_blocks;
    h_hllDevice.blocks = d_blocks;
    
    HLLMatrixDevice *d_hll;
    cudaMalloc((void**)&d_hll, sizeof(HLLMatrixDevice));
    cudaMemcpy(d_hll, &h_hllDevice, sizeof(HLLMatrixDevice), cudaMemcpyHostToDevice);

    double *d_x;
    cudaMalloc((void**)&d_x, mat->N * sizeof(double));
    cudaMemcpy(d_x, x, mat->N * sizeof(double), cudaMemcpyHostToDevice);
    
    // Il vettore y globale ha dimensione num_blocks * HACKSIZE (alcune righe potrebbero non essere usate nell'ultimo blocco)
    int total_rows = num_blocks * HACKSIZE;
    double *d_y;
    cudaMalloc((void**)&d_y, total_rows * sizeof(double));

    // ===================
    // Lancio del kernel
    // ===================
    spmv_hll_cuda(d_hll, d_x, d_y);
    
    // Copia del risultato dal device all'host e stampa
    double *h_y = (double *) malloc(total_rows * sizeof(double));
    cudaMemcpy(h_y, d_y, total_rows * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("Risultato del prodotto matrice-vettore:\n");
    // Stampo soltanto le righe "valide" di ciascun blocco
    for (int b = 0; b < num_blocks; b++) {
        int block_rows = hll.blocks[b].rows;
        for (int i = 0; i < block_rows; i++) {
            int global_row = b * HACKSIZE + i;
            printf("y[%d] = %f\n", global_row, h_y[global_row]);
        }
    }

    // ===================
    // Pulizia della memoria Device
    // ===================
    for (int b = 0; b < num_blocks; b++) {
        cudaFree(h_blockDevices[b].JA_flat);
        cudaFree(h_blockDevices[b].AS_flat);
    }
    free(h_blockDevices);
    cudaFree(d_blocks);
    cudaFree(d_hll);
    cudaFree(d_x);
    cudaFree(d_y);
    

    // ===================
    // Pulizia della memoria Host
    // ===================

    freeHLLMatrix(&hll);
    free(x);
    free(h_y);


    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(res_csr_cuda);

    return 0;
}
