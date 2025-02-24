#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../lib/utils.h"

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }    

    MatrixElement *mat = read_matrix(argv[1]);
    if (!mat) exit(1);

    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;
    char *matrix_name;

    matrix_name = strrchr(argv[1], '/');
    if (matrix_name != NULL) {
        matrix_name++; 
    } else {
        matrix_name = argv[1];
    }

    char *dot = strrchr(matrix_name, '.');
    if (dot != NULL) {
        *dot = '\0';
    }
    
    //Creazione vettore x
    double *x = generateVector(cols);

    //Creazione del mediatore
    MatrixConversionMediator mediator = createMatrixMediator();

    //Variabili per misura delle prestazioni
    clock_t start_time, end_time;
    double times[REPETITIONS];
    int i;

    double *time_serial = (double *) malloc(sizeof(double));
    if(time_serial == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    double *time_csr = (double *) malloc(sizeof(double));
    if(time_csr == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    double *time_hll = (double *) malloc(sizeof(double));
    if(time_hll == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    //Creazione struct formato CSR
    CSRMatrix csr;

    //Conversione formato CSR
    mediator.convertToCSR(mat->matrix, nz, rows, &csr);

    //Allocazione risultato seriale
    double *y_serial = (double *)calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo seriale e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = clock();
        prodSerial(rows, &csr, x, y_serial);   
        end_time = clock();
        double elapsed_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;
        times[i] = elapsed_time;
    }

    printResult(y_serial, rows);

    calculatePerformance(times, mat, matrix_name, "serial", "cuda", 0, time_serial);
    
    memset(times, 0, sizeof(times));

    //Allocazione risultato CUDA CSR
    double *y_csr = (double *)calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    float *elapsed_time_csr = (float *) malloc(sizeof(float));
    if(elapsed_time_csr == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < REPETITIONS; i++) {
        prodCudaCSR(rows, cols, csr.IRP, csr.JA, csr.AS, x, y_csr, elapsed_time_csr);  
        times[i] = *elapsed_time_csr;
    }

    if(checkResults(y_serial, y_csr, rows) == 0){
        printf("Serial result is different from parallel result with csr\n");
        exit(1);
    }

    printf("CSR results checked\n");

    calculatePerformance(times, mat, matrix_name, "CSR", "cuda", 0, time_csr);

    free(y_csr);
    free(elapsed_time_csr);
    freeCSRMatrix(&csr);

    memset(times, 0, sizeof(times));

    // Creazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);
    trasponseHLLMatrix(&hll);

    //Allocazione risultato Openmp HLL 
    double *y_hll = (double *)calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    float *elapsed_time_hll = (float *) malloc(sizeof(float));
    if(elapsed_time_hll == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < REPETITIONS; i++) {
        prodCudaHLL(&hll, mat->M, mat->N, x, y_hll, elapsed_time_hll); 
        times[i] = *elapsed_time_hll;
    }

    if(checkResults(y_serial, y_hll, rows) == 0){
        printf("Serial result is different from parallel result with hll\n");
        exit(1);
    }

    printf("HLL results checked\n");

    calculatePerformance(times, mat, matrix_name, "HLL", "cuda", 0, time_hll);

    calculateSpeedup(matrix_name, *time_serial, *time_csr, *time_hll, "cuda", 0);

    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(elapsed_time_hll);
    free(y_serial);
    free(y_hll);
    freeHLLMatrix(&hll);
    free(x);
    free(mat->matrix);
    free(mat);

    return 0;
}