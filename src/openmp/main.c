#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../../lib/utils.h"

void run_single_execution(char *matrix_name, MatrixElement *mat) {
    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;
    
    //Creazione vettore x
    double *x = generate_vector(matrix_name, cols);

    //Variabili per misura delle prestazioni
    double start_time, end_time = 0.0;
    double times[REPETITIONS];
    int i;
    char *filename_p = "results/openmp/performance.csv";
    char *filename_s = "results/openmp/speedup.csv";


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
    double *y_serial = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo seriale e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_serial(rows, csr, x, y_serial);
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    // printResult(y_serial, rows);

    calculate_performance_openmp(times, mat, matrix_name, "serial", 1, time_serial, filename_p);
    
    memset(times, 0, sizeof(times));
    start_time = end_time = 0.0;

    //Allocazione risultato OpenMP CSR
    double *y_csr = calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    int num_threads = omp_get_max_threads();
    int *row_bounds = malloc((num_threads + 1) * sizeof(int));
    if (row_bounds == NULL) {
        fprintf(stderr, "Errore di allocazione per row_bounds.\n");
        exit(EXIT_FAILURE);
    }

    compute_row_bounds(csr, rows, num_threads, row_bounds);

    //Calcolo parallelo openmp CSR e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_openmp_csr(rows, csr, x, y_csr, row_bounds);
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    if(check_results(y_serial, y_csr, rows) == 0){
        printf("Serial result is different from parallel result with csr\n");
        exit(1);
    }

    printf("CSR results checked\n");

    calculate_performance_openmp(times, mat, matrix_name, "CSR", omp_get_max_threads(), time_csr, filename_p);

    free(y_csr);
    free_csr_matrix(csr);

    memset(times, 0, sizeof(times));
    start_time = end_time = 0.0;

    // Creazione struct formato HLL
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE);

    //Allocazione risultato Openmp HLL 
    double *y_hll = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo parallelo openmp HLL e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_openmp_hll(hll, x, y_hll);
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    if(check_results(y_serial, y_hll, rows) == 0){
        printf("Serial result is different from parallel result with hll\n");
        exit(1);
    }

    printf("HLL results checked\n");

    calculate_performance_openmp(times, mat, matrix_name, "HLL", omp_get_max_threads(), time_hll, filename_p);

    calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, filename_s , omp_get_max_threads(), nz);

    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(y_serial);
    free(y_hll);
    free_hll_matrix(hll);
    free(x);
    free(mat->matrix);
    free(mat);
    free(row_bounds);
}


void run_all_threads_execution(char *matrix_name, MatrixElement *mat){
    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;

    //Creazione vettore x
    double *x = generate_vector(matrix_name, cols);

    //Variabili per misura delle prestazioni
    double start_time, end_time = 0.0;
    double times[REPETITIONS];
    int i;
    int num_threads;
    char *filename_p = "results/openmp/performance_threads.csv";
    char *filename_s = "results/openmp/speedup_threads.csv";

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
    // Creazione struct formato HLL
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE);

    //Allocazione risultato seriale
    double *y_serial = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Allocazione risultato OpenMP CSR
    double *y_csr = calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Allocazione risultato Openmp HLL 
    double *y_hll = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    //Calcolo seriale e misura dei tempi
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_serial(rows, csr, x, y_serial);
        end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        times[i] = elapsed_time;
    }

    // printResult(y_serial, rows);

    calculate_performance_openmp(times, mat, matrix_name, "serial", 1, time_serial, filename_p);
    
    memset(times, 0, sizeof(times));
    start_time = end_time = 0.0;

    for (num_threads = 1; num_threads <= 40; num_threads++) {

        omp_set_num_threads(num_threads);
        printf("Number of threads: %d\n", num_threads);

        int *row_bounds = malloc((num_threads + 1) * sizeof(int));
        if (row_bounds == NULL) {
            fprintf(stderr, "Errore di allocazione per row_bounds.\n");
            exit(EXIT_FAILURE);
        }

        compute_row_bounds(csr, rows, num_threads, row_bounds);

        //Calcolo parallelo openmp CSR e misura dei tempi
        for (i = 0; i < REPETITIONS; i++) {
            start_time = omp_get_wtime();
            prod_openmp_csr(rows, csr, x, y_csr, row_bounds);
            end_time = omp_get_wtime();
            double elapsed_time = end_time - start_time;
            times[i] = elapsed_time;
        }

        if(check_results(y_serial, y_csr, rows) == 0){
            printf("Serial result is different from parallel result with csr\n");
            exit(1);
        }

        printf("CSR results checked\n");

        calculate_performance_openmp(times, mat, matrix_name, "CSR", omp_get_max_threads(), time_csr, filename_p);

        memset(times, 0, sizeof(times));
        start_time = end_time = 0.0;

        //Calcolo parallelo openmp HLL e misura dei tempi
        for (i = 0; i < REPETITIONS; i++) {
            start_time = omp_get_wtime();
            prod_openmp_hll(hll, x, y_hll);
            end_time = omp_get_wtime();
            double elapsed_time = end_time - start_time;
            times[i] = elapsed_time;
        }

        if(check_results(y_serial, y_hll, rows) == 0){
            printf("Serial result is different from parallel result with hll\n");
            exit(1);
        }

        printf("HLL results checked\n");

        calculate_performance_openmp(times, mat, matrix_name, "HLL", omp_get_max_threads(), time_hll, filename_p);

        calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, filename_s, omp_get_max_threads(), nz);
        
        free(row_bounds);
    }


    free(y_csr);
    free_csr_matrix(csr);
    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(y_serial);
    free(y_hll);
    free_hll_matrix(hll);
    free(x);
    free(mat->matrix);
    free(mat);

}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [mode 0-1]\n", argv[0]);
        exit(1);
    }    

    MatrixElement *mat = read_matrix(argv[1]);
    if (!mat) exit(1);

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

    int mode = atoi(argv[2]);

    if (mode == 0){
        run_single_execution(matrix_name, mat);
    } else {
        run_all_threads_execution(matrix_name, mat);
    }

    return 0;
}
