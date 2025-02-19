#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../../lib/utils.h"

void serialExecutionOpenmp(MatrixElement *mat, MatrixConversionMediator mediator){
    printf("Esecuzione seriale\n");
    double start_time, end_time, total_time = 0.0;
    double times[REPETITIONS];

    double *x = generateVector(mat->N);
    double *y = calloc(mat->M, sizeof(double));  // Allocazione del risultato
    if (!y) {
        fprintf(stderr, "Errore di allocazione della memoria\n");
        exit(1);
    }

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);

    for (int i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prodSerial(mat->M, &csr, x, y);   
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    //calcolo tempi

    printResult(y, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(y);

}

void hllExecutionOpenmp(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione HLL Openmp\n");

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
    prodOpenmpHLL(&hll, x, y);  

    printResult(y, mat->M);

    free(x);
    freeHLLMatrixCuda(&hll);
    free(mat->matrix);
    free(mat);
    free(y);

}

void csrExecutionOpenmp(MatrixElement *mat, MatrixConversionMediator mediator){

    printf("Esecuzione CSR Openmp\n");

    double *x = generateVector(mat->N);
    double *y = calloc(mat->M, sizeof(double));  // Allocazione del risultato
    if (!y) {
        fprintf(stderr, "Errore di allocazione della memoria\n");
        exit(1);
    }

    // Creazione struct formato CSR
    CSRMatrix csr;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    
    
    prodOpenmpCSR(mat->M, &csr, x, y);  

    printResult(y, mat->M);

    free(x);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(y);

}


