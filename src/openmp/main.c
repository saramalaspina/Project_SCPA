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

    char *dot = strrchr(matrix_name, '.');
    if (dot != NULL) {
        *dot = '\0';
    }

    //Creazione vettore x
    double *x = generateVector(cols);

    //Variabili per misura delle prestazioni
    double start_time, end_time = 0.0;
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

    qsort(mat->matrix, nz, sizeof(COOElement), compareCOO);

    //Creazione struct formato CSR
    CSRMatrix *csr = convertCOOtoCSR(mat->matrix, nz, rows);

    //Allocazione risultato seriale
    double *y_serial = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo seriale e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prodSerial(rows, csr, x, y_serial);   
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    // printResult(y_serial, rows);

    calculatePerformanceOpenMP(times, mat, matrix_name, "serial", 1, time_serial);
    
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
        prodOpenmpCSR(rows, csr, x, y_csr);   
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    if(checkResults(y_serial, y_csr, rows) == 0){
        printf("Serial result is different from parallel result with csr\n");
        exit(1);
    }

    printf("CSR results checked\n");

    calculatePerformanceOpenMP(times, mat, matrix_name, "CSR", omp_get_max_threads(), time_csr);

    free(y_csr);
    freeCSRMatrix(csr);

    memset(times, 0, sizeof(times));
    start_time = end_time = 0.0;

    // Creazione struct formato HLL
    HLLMatrix *hll = convertCOOtoHLL(mat, HACKSIZE);

    //Allocazione risultato Openmp HLL 
    double *y_hll = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo parallelo openmp HLL e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prodOpenmpHLL(hll, x, y_hll);  
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    if(checkResults(y_serial, y_hll, rows) == 0){
        printf("Serial result is different from parallel result with hll\n");
        exit(1);
    }

    printf("HLL results checked\n");

    calculatePerformanceOpenMP(times, mat, matrix_name, "HLL", omp_get_max_threads(), time_hll);

    calculateSpeedup(matrix_name, *time_serial, *time_csr, *time_hll, "openmp", omp_get_max_threads());

    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(y_serial);
    free(y_hll);
    freeHLLMatrix(hll);
    free(x);
    free(mat->matrix);
    free(mat);

    return 0;
}
