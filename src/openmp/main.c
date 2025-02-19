#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

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

    //Creazione vettore x
    double *x = generateVector(cols);

    //Creazione del mediatore
    MatrixConversionMediator mediator = createMatrixMediator();

    //Variabili per misura delle prestazioni
    double start_time, end_time = 0.0;
    double times[REPETITIONS];
    int i;

    //Creazione struct formato CSR
    CSRMatrix csr;

    //Conversione formato CSR
    mediator.convertToCSR(mat->matrix, nz, rows, &csr);

    //Allocazione risultato seriale
    double *y_serial = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo seriale e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prodSerial(rows, &csr, x, y_serial);   
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    printResult(y_serial, rows);

    calculatePerformance(times, mat, matrix_name, "serial", "openmp", 1);
    
    memset(times, 0, sizeof(times));
    start_time = end_time = 0.0;

    //Allocazione risultato OpenMP CSR
    double *y_csr = calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo parallelo openmp CSR e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prodOpenmpCSR(rows, &csr, x, y_csr);   
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    if(checkResults(y_serial, y_csr, rows) == 0){
        printf("Serial result is different from parallel result with csr\n");
        exit(1);
    }

    printf("CSR results checked\n");

    calculatePerformance(times, mat, matrix_name, "CSR", "openmp", omp_get_max_threads());

    free(y_csr);
    freeCSRMatrix(&csr);

    memset(times, 0, sizeof(times));
    start_time = end_time = 0.0;

    // Creazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);
    trasponseHLLMatrix(&hll);

    //Allocazione risultato Openmp HLL 
    double *y_hll = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo parallelo openmp HLL e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prodOpenmpHLL(&hll, x, y_hll);  
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    if(checkResults(y_serial, y_hll, rows) == 0){
        printf("Serial result is different from parallel result with hll\n");
        exit(1);
    }

    printf("HLL results checked\n");

    calculatePerformance(times, mat, matrix_name, "HLL", "openmp", omp_get_max_threads());

    free(y_serial);
    free(y_hll);
    freeHLLMatrix(&hll);
    free(x);
    free(mat->matrix);
    free(mat);

    return 0;
}
