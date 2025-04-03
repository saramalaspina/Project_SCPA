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
    double *x = generate_vector(matrix_name, cols);

    //Variabili per misura delle prestazioni
    clock_t start_time, end_time;
    int i;

    double *times = (double *)malloc(REPETITIONS * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

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

    qsort(mat->matrix, nz, sizeof(COOElement), compare_coo);
    
    //Creazione struct formato CSR
    CSRMatrix *csr = convert_coo_to_csr(mat->matrix, nz, rows);

    //Allocazione risultato seriale
    double *y_serial = (double *)calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo seriale e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = clock();
        prod_serial(rows, csr, x, y_serial);
        end_time = clock();
        times[i] = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000;
    }

    // printResult(y_serial, rows);

    calculate_performance_cuda(times, mat, matrix_name, "serial", time_serial);
    
    memset(times, 0, REPETITIONS * sizeof(double));

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
        prod_cuda_csr(rows, cols, csr, x, y_csr, elapsed_time_csr);
        times[i] = *elapsed_time_csr;
    }

    if(check_results(y_serial, y_csr, rows) == 0){
        printf("Serial result is different from parallel result with csr\n");
    } else {
        printf("CSR results checked\n");
    }

    calculate_performance_cuda(times, mat, matrix_name, "CSR", time_csr);

    memset(times, 0, REPETITIONS * sizeof(double));

    free(y_csr);
    free(elapsed_time_csr);
    free_csr_matrix(csr);

    // Creazione struct formato HLL
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE) ;

    //Allocazione risultato Openmp HLL 
    double *y_hll = (double *)calloc(rows, sizeof(double)); 
    if (!y_hll) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    float *elapsed_time_hll = (float *) malloc(sizeof(float));
    if(elapsed_time_hll == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    
    for (i = 0; i < REPETITIONS; i++) {
        prod_cuda_hll(hll, x, y_hll, rows, elapsed_time_hll);
        times[i] = *elapsed_time_hll;
    }

    if(check_results(y_serial, y_hll, rows) == 0){
        printf("Serial result is different from parallel result with hll\n");
    } else {
        printf("HLL results checked\n");
    }

    calculate_performance_cuda(times, mat, matrix_name, "HLL", time_hll);

    calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, "results/cuda/speedup.csv", 0);

    free(times);
    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(elapsed_time_hll);
    free(y_serial);
    free(y_hll);
    free_hll_matrix(hll);
    free(x);
    free(mat->matrix);
    free(mat);

    return 0;
}
