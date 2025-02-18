#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

void serialExecutionOpenmp(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione seriale\n");

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

void hllExecutionOpenmp(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione HLL Openmp\n");

    double *x = generateVector(mat->N);

    // Creazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);
    trasponseHLLMatrix(&hll);

    // **Esecuzione su GPU (HLL)**
    double *res = prodOpenmpHLL(&hll, x);  

    printResult(res, mat->M);

    free(x);
    freeHLLMatrixCuda(&hll);
    free(mat->matrix);
    free(mat);
    free(res);

}

void csrExecutionOpenmp(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione CSR Openmp\n");

    double *x = generateVector(mat->N);

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    
    // **Esecuzione su GPU (CSR)**
    double *res = prodOpenmpCSR(mat->M, &csr, x);  

    printResult(res, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(res);

}


