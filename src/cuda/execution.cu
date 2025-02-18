#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

void serialExecutionCuda(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione Seriale\n");

    double *x = generateVector(mat->N);

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    
    double *res = prodSerial(mat->M, &csr, x);    

    printResult(res, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(res);

}

void hllExecutionCuda(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione HLL Cuda\n");

    double *x = generateVector(mat->N);

    // Creazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);
    trasponseHLLMatrix(&hll);

    // **Esecuzione su GPU (HLL)**
    double *res = prodCudaHLL(&hll, mat->M, mat->N, x);  

    printResult(res, mat->M);

    free(x);
    freeHLLMatrixCuda(&hll);
    free(mat->matrix);
    free(mat);
    free(res);

}

void csrExecutionCuda(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione CSR Cuda\n");

    double *x = generateVector(mat->N);

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    
    // **Esecuzione su GPU (CSR)**
    double *res = prodCudaCSR(mat->M, mat->N, csr.IRP, csr.JA, csr.AS, x);  

    printResult(res, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(res);

}