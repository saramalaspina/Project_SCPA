#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../lib/utils.h"

void run_test_execution(char *matrix_name, MatrixElement *mat) {
    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;

    int values_block[] = {128, 256, 512, 1024};

    // Generate input vector x
    double *x = generate_vector(cols);

    // Performance measurement variables
    clock_t start_time, end_time;
    int i;

    // Allocate memory for storing multiple execution times
    double *times = (double *)malloc(REPETITIONS * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for average execution time of each version
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

    double *time_csr_warp = (double *) malloc(sizeof(double));
    if(time_csr == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    double *time_hll_warp = (double *) malloc(sizeof(double));
    if(time_hll == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
  
    // Sort the matrix in COO format
    qsort(mat->matrix, nz, sizeof(COOElement), compare_coo);
    
    // Convert matrix from COO to CSR format
    CSRMatrix *csr = convert_coo_to_csr(mat->matrix, nz, rows);
    // Convert matrix from COO to HLL format
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE);

    // Allocate output vector for the serial result
    double *y_serial = (double *)calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Run the serial version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = clock();
        prod_serial(rows, csr, x, y_serial);
        end_time = clock();
        times[i] = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000;
    }

    calculate_performance_cuda(times, mat, matrix_name, "serial", time_serial, "results/cuda/performance.csv", 0);
    
    // Reset the times array
    memset(times, 0, REPETITIONS * sizeof(double));

    // Allocate output vector for CUDA CSR result
    double *y_csr = (double *)calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    } 

    // Allocate memory for CUDA elapsed time
    float *elapsed_time_csr = (float *) malloc(sizeof(float));
    if(elapsed_time_csr == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }  

     // Allocate output vector for CUDA HLL result
    double *y_hll = (double *)calloc(rows, sizeof(double)); 
    if (!y_hll) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    } 
    
    // Allocate memory for CUDA HLL elapsed time
    float *elapsed_time_hll = (float *) malloc(sizeof(float));
    if(elapsed_time_hll == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }    

    for (int j = 0; j < 4; j++) {
        int threads_per_block = values_block[j];
        
        // Run CUDA CSR version and measure execution time
        for (i = 0; i < REPETITIONS; i++) {
            prod_cuda_csr(rows, cols, csr, x, y_csr, elapsed_time_csr, threads_per_block);
            times[i] = *elapsed_time_csr;
        }

        // Verify correctness of CSR CUDA result
        if(check_results(y_serial, y_csr, rows) == 0){
            printf("Serial result is different from parallel result with csr\n");
        } else {
            printf("CSR results checked\n");
        }

        // Calculate average time for CUDA CSR
        calculate_performance_cuda(times, mat, matrix_name, "CSR", time_csr, "results/cuda/performance.csv", threads_per_block);

        // Reset times array
        memset(times, 0, REPETITIONS * sizeof(double));


        // Run CUDA CSR version with warp and measure execution time
        for (i = 0; i < REPETITIONS; i++) {
            prod_cuda_csr_warp(rows, cols, csr, x, y_csr, elapsed_time_csr, threads_per_block);
            times[i] = *elapsed_time_csr;
        }
        
        // Verify correctness of CSR CUDA result
        if(check_results(y_serial, y_csr, rows) == 0){
            printf("Serial result is different from parallel result with csr (warp)\n");
        } else {
            printf("CSR (warp) results checked\n");
        }
        
        // Calculate average time for CUDA CSR
        calculate_performance_cuda(times, mat, matrix_name, "CSR", time_csr_warp, "results/cuda/performance_warp.csv", threads_per_block);
        
        // Reset times array
        memset(times, 0, REPETITIONS * sizeof(double));
        
        // Run CUDA HLL version and measure execution time
        for (i = 0; i < REPETITIONS; i++) {
            prod_cuda_hll(hll, x, y_hll, rows, elapsed_time_hll, threads_per_block);
            times[i] = *elapsed_time_hll;
        }

        // Verify correctness of HLL CUDA result
        if(check_results(y_serial, y_hll, rows) == 0){
            printf("Serial result is different from parallel result with hll\n");
        } else {
            printf("HLL results checked\n");
        }

        calculate_performance_cuda(times, mat, matrix_name, "HLL", time_hll, "results/cuda/performance.csv", threads_per_block);

        // Reset times array
        memset(times, 0, REPETITIONS * sizeof(double));

        // Run CUDA HLL version with warp and measure execution time
        for (i = 0; i < REPETITIONS; i++) {
            prod_cuda_hll_warp(hll, x, y_hll, rows, elapsed_time_hll, threads_per_block);
            times[i] = *elapsed_time_hll;
        }

        // Verify correctness of HLL CUDA result
        if(check_results(y_serial, y_hll, rows) == 0){
            printf("Serial result is different from parallel result with hll (warp)\n");
        } else {
            printf("HLL (warp) results checked\n");
        }

        calculate_performance_cuda(times, mat, matrix_name, "HLL", time_hll_warp, "results/cuda/performance_warp.csv", threads_per_block);

        // Compute and save speedup results
        printf("Calculate speedup\n");
        calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, "results/cuda/speedup.csv", threads_per_block, nz);
        printf("Calculate speedup (warp)\n");
        calculate_speedup(matrix_name, *time_serial, *time_csr_warp, *time_hll_warp, "results/cuda/speedup_warp.csv", threads_per_block, nz);
    }

    // Free all allocated memory
    free(y_csr);
    free(elapsed_time_csr);
    free_csr_matrix(csr);
    free(times);
    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(time_csr_warp);
    free(time_hll_warp);
    free(elapsed_time_hll);
    free(y_serial);
    free(y_hll);
    free_hll_matrix(hll);
    free(x);
    free(mat->matrix);
    free(mat);
}

void run_best_execution(char *matrix_name, MatrixElement *mat) {
    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;
    
    // Generate input vector x
    double *x = generate_vector(cols);

    // Performance measurement variables
    clock_t start_time, end_time;
    int i;

    // Allocate memory for storing multiple execution times
    double *times = (double *)malloc(REPETITIONS * sizeof(double));
    if (times == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for average execution time of each version
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

    // Sort the matrix in COO format
    qsort(mat->matrix, nz, sizeof(COOElement), compare_coo);
    
    // Convert matrix from COO to CSR format
    CSRMatrix *csr = convert_coo_to_csr(mat->matrix, nz, rows);

    // Allocate output vector for the serial result
    double *y_serial = (double *)calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Run the serial version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = clock();
        prod_serial(rows, csr, x, y_serial);
        end_time = clock();
        times[i] = ((double)(end_time - start_time) / CLOCKS_PER_SEC) * 1000;
    }

    calculate_performance_cuda(times, mat, matrix_name, "serial", time_serial, "results/cuda/best_performance.csv", 0);
    
    // Reset the times array
    memset(times, 0, REPETITIONS * sizeof(double));

    // Allocate output vector for CUDA CSR result
    double *y_csr = (double *)calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Allocate memory for CUDA elapsed time
    float *elapsed_time_csr = (float *) malloc(sizeof(float));
    if(elapsed_time_csr == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }


    // Run CUDA CSR version with warp and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        prod_cuda_csr_warp(rows, cols, csr, x, y_csr, elapsed_time_csr, THREADS_PER_BLOCK);
        times[i] = *elapsed_time_csr;
    }
    
    // Verify correctness of CSR CUDA result
    if(check_results(y_serial, y_csr, rows) == 0){
        printf("Serial result is different from parallel result with csr (warp)\n");
    } else {
        printf("CSR (warp) results checked\n");
    }
    
    // Calculate average time for CUDA CSR
    calculate_performance_cuda(times, mat, matrix_name, "CSR", time_csr, "results/cuda/best_performance.csv", THREADS_PER_BLOCK);
    
    // Reset times array
    memset(times, 0, REPETITIONS * sizeof(double));

    // Free CSR-related resources
    free(y_csr);
    free(elapsed_time_csr);
    free_csr_matrix(csr);

    // Convert matrix from COO to HLL format
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE);

    // Allocate output vector for CUDA HLL result
    double *y_hll = (double *)calloc(rows, sizeof(double)); 
    if (!y_hll) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Allocate memory for CUDA HLL elapsed time
    float *elapsed_time_hll = (float *) malloc(sizeof(float));
    if(elapsed_time_hll == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    
    // Run CUDA HLL version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        prod_cuda_hll(hll, x, y_hll, rows, elapsed_time_hll, THREADS_PER_BLOCK);
        times[i] = *elapsed_time_hll;
    }

    // Verify correctness of HLL CUDA result
    if(check_results(y_serial, y_hll, rows) == 0){
        printf("Serial result is different from parallel result with hll\n");
    } else {
        printf("HLL results checked\n");
    }

    calculate_performance_cuda(times, mat, matrix_name, "HLL", time_hll, "results/cuda/best_performance.csv", THREADS_PER_BLOCK);

    // Compute and save speedup results
    printf("Calculate speedup\n");
    calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, "results/cuda/best_speedup.csv", THREADS_PER_BLOCK, nz);

    // Free all allocated memory
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
}



// Main function to load matrix and select execution mode
int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s [matrix-market-filename] [mode 0-1]\n", argv[0]);
        exit(1);
    }    

    // Read the matrix from a Matrix Market file
    MatrixElement *mat = read_matrix(argv[1]);
    if (!mat) exit(1);

    // Extract matrix name from the file path
    char *matrix_name = strrchr(argv[1], '/');
    if (matrix_name != NULL) {
        matrix_name++; 
    } else {
        matrix_name = argv[1];
    }

    char *dot = strrchr(matrix_name, '.');
    if (dot != NULL) {
        *dot = '\0';
    }

    // Parse mode and run the appropriate function
    int mode = atoi(argv[2]);
    if (mode == 0){
        run_test_execution(matrix_name, mat);
    } else {
        run_best_execution(matrix_name, mat);
    }

    return 0;
}
