#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

void serialExecutionCuda(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione Seriale\n");

    double *x = generateVector(mat->N);
    double *y = calloc(mat->M, sizeof(double));  // Allocazione del risultato
    if (!y) {
        fprintf(stderr, "Errore di allocazione della memoria\n");
        exit(1);
    }

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    
    prodSerial(mat->M, &csr, x, y);    

    printResult(y, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(y);

}

void hllExecutionCuda(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione HLL Cuda\n");

    double *x = generateVector(mat->N);
    double *y = calloc(mat->M, sizeof(double));  // Allocazione del risultato
    if (!y) {
        fprintf(stderr, "Errore di allocazione della memoria\n");
        exit(1);
    }


    // Creazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);
    trasponseHLLMatrix(&hll);

    // **Esecuzione su GPU (HLL)**
    prodCudaHLL(&hll, mat->M, mat->N, x, y);  

    printResult(y, mat->M);

    free(x);
    freeHLLMatrixCuda(&hll);
    free(mat->matrix);
    free(mat);
    free(y);

}

void csrExecutionCuda(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione CSR Cuda\n");

    double *x = generateVector(mat->N);
    double *y = calloc(mat->M, sizeof(double));  // Allocazione del risultato
    if (!y) {
        fprintf(stderr, "Errore di allocazione della memoria\n");
        exit(1);
    }

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    
    // **Esecuzione su GPU (CSR)**
    prodCudaCSR(mat->M, mat->N, csr.IRP, csr.JA, csr.AS, x, y);  

    printResult(y, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(y);

}