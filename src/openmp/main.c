#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "../../lib/utils.h" 

// Function to execute a single run of the matrix-vector product 
void run_single_execution(char *matrix_name, MatrixElement *mat) {
    int num_threads = omp_get_max_threads();

    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;

    // Generate input vector x
    double *x = generate_vector(cols);

    // Performance measurement variables
    double start_time, end_time = 0.0;
    double times[REPETITIONS];
    int i;

    // Allocate memory for storing performance times
    double *time_serial = (double *) malloc(sizeof(double));
    double *time_csr = (double *) malloc(sizeof(double));
    double *time_hll = (double *) malloc(sizeof(double));
    double *time_csr_guided = (double *) malloc(sizeof(double));
    double *time_hll_guided = (double *) malloc(sizeof(double));
    if (!time_serial || !time_csr || !time_hll || !time_csr_guided || !time_hll_guided) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Sort the matrix in COO format
    qsort(mat->matrix, nz, sizeof(COOElement), compare_coo);

    // Convert matrix from COO to CSR format
    CSRMatrix *csr = convert_coo_to_csr(mat->matrix, nz, rows);

    // Allocate output vector for the serial result
    double *y_serial = calloc(rows, sizeof(double)); 
    if (!y_serial) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Run the serial version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_serial(rows, csr, x, y_serial);
        end_time = omp_get_wtime();
        times[i] = end_time - start_time;
    }

    calculate_performance_openmp(times, mat, matrix_name, "serial", 1, time_serial, "results/openmp/performance.csv");
    
    // Reset the times array
    memset(times, 0, sizeof(times));

    // Allocate output vector for OpenMP CSR result
    double *y_csr = calloc(rows, sizeof(double)); 
    if (!y_csr) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Allocate and compute thread workload boundaries
    int *row_bounds = malloc((num_threads + 1) * sizeof(int));
    if (row_bounds == NULL) {
        fprintf(stderr, "Allocation error for row_bounds.\n");
        exit(EXIT_FAILURE);
    }

    start_time = omp_get_wtime();
    compute_row_bounds(csr, rows, num_threads, row_bounds);
    end_time = omp_get_wtime();
    double pre_time_csr = (end_time - start_time)*1000;

    // Run the OpenMP CSR version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_openmp_csr(rows, csr, x, y_csr, row_bounds);
        end_time = omp_get_wtime();
        times[i] = end_time - start_time;
    }

    // Verify correctness of CSR OpenMP result
    if (check_results(y_serial, y_csr, rows) == 0) {
        printf("Serial result is different from parallel result with csr\n");
    } else {
        printf("CSR results checked\n");
    }

    calculate_performance_openmp(times, mat, matrix_name, "CSR", num_threads, time_csr, "results/openmp/performance.csv");

    memset(times, 0, sizeof(times));

    // Run the OpenMP CSR guided version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_openmp_csr_guided(csr, x, y_csr, rows);
        end_time = omp_get_wtime();
        times[i] = end_time - start_time;
    }

    // Verify correctness of CSR OpenMP result
    if (check_results(y_serial, y_csr, rows) == 0) {
        printf("Serial result is different from parallel result with csr (guided)\n");
    } else {
        printf("CSR (guided) results checked\n");
    }

    calculate_performance_openmp(times, mat, matrix_name, "CSR", num_threads, time_csr_guided, "results/openmp/performance_guided.csv");    

    // Free CSR-related resources
    free(y_csr);
    free_csr_matrix(csr);
    free(row_bounds);

    // Reset the times array
    memset(times, 0, sizeof(times));

    // Convert matrix from COO to HLL format
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE);

    int *block_bounds = malloc((num_threads + 1) * sizeof(int));
    if (block_bounds == NULL) {
        fprintf(stderr, "Allocation error for block_bounds.\n");
        exit(EXIT_FAILURE);
    }

    start_time = omp_get_wtime();
    compute_block_bounds(hll->numBlocks, num_threads, block_bounds);
    end_time = omp_get_wtime();
    double pre_time_hll = (end_time - start_time)*1000;

    // Allocate output vector for OpenMP HLL result
    double *y_hll = calloc(rows, sizeof(double)); 
    if (!y_hll) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Run the OpenMP HLL version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_openmp_hll(hll, x, y_hll, block_bounds);
        end_time = omp_get_wtime();
        times[i] = end_time - start_time;
    }

    // Verify correctness of HLL OpenMP result
    if (check_results(y_serial, y_hll, rows) == 0) {
        printf("Serial result is different from parallel result with hll\n");
    } else {
        printf("HLL results checked\n");
    }

    calculate_performance_openmp(times, mat, matrix_name, "HLL", num_threads, time_hll, "results/openmp/performance.csv");

    // Reset the times array
    memset(times, 0, sizeof(times));

    // Run the OpenMP HLL guided version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_openmp_hll_guided(hll, x, y_hll);
        end_time = omp_get_wtime();
        times[i] = end_time - start_time;
    }

    // Verify correctness of HLL OpenMP result
    if (check_results(y_serial, y_hll, rows) == 0) {
        printf("Serial result is different from parallel result with hll (guided)\n");
    } else {
        printf("HLL (guided) results checked\n");
    }

    calculate_performance_openmp(times, mat, matrix_name, "HLL", num_threads, time_hll_guided, "results/openmp/performance_guided.csv");

    // Compute and save speedup results
    printf("Calculate speedup\n");
    calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, "results/openmp/speedup.csv", num_threads, nz);
    printf("Calculate speedup (guided)\n");
    calculate_speedup(matrix_name, *time_serial, *time_csr_guided, *time_hll_guided, "results/openmp/speedup_guided.csv", num_threads, nz);

    preprocessing_performance_openmp(matrix_name, nz, "CSR", pre_time_csr, *time_csr, *time_csr_guided, num_threads);
    preprocessing_performance_openmp(matrix_name, nz, "HLL", pre_time_hll, *time_hll, *time_hll_guided, num_threads);

    // Free all allocated memory
    free(time_serial);
    free(time_csr);
    free(time_hll);
    free(y_serial);
    free(y_hll);
    free_hll_matrix(hll);
    free(x);
    free(mat->matrix);
    free(mat);
    free(block_bounds);
}

// Function to execute multiple run of the matrix-vector product varying the number of threads from 1 to 40
void run_all_threads_execution(char *matrix_name, MatrixElement *mat) {
    int rows = mat->M;
    int cols = mat->N;
    int nz = mat->nz;

    char *filename_p = "results/openmp/performance_threads.csv";
    char *filename_s = "results/openmp/speedup_threads.csv";

    // Generate input vector x
    double *x = generate_vector(cols);

    // Performance measurement variables
    double start_time, end_time = 0.0;
    double times[REPETITIONS];
    int i, num_threads;

    // Allocate memory for storing performance times
    double *time_serial = malloc(sizeof(double));
    double *time_csr = malloc(sizeof(double));
    double *time_hll = malloc(sizeof(double));
    if (!time_serial || !time_csr || !time_hll) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }

    // Sort the matrix in COO format
    qsort(mat->matrix, nz, sizeof(COOElement), compare_coo);

    // Convert matrix from COO to CSR format
    CSRMatrix *csr = convert_coo_to_csr(mat->matrix, nz, rows);
    // Convert matrix from COO to HLL format
    HLLMatrix *hll = convert_coo_to_hll(mat, HACKSIZE);

    // Allocate output vectors for serial, OpenMP CSR and OpenMP HLL result
    double *y_serial = calloc(rows, sizeof(double)); 
    double *y_csr = calloc(rows, sizeof(double)); 
    double *y_hll = calloc(rows, sizeof(double)); 
    if (!y_serial || !y_csr || !y_hll) {
        fprintf(stderr, "Memory allocation error\n");
        exit(1);
    }

    // Run the serial version and measure execution time
    for (i = 0; i < REPETITIONS; i++) {
        start_time = omp_get_wtime();
        prod_serial(rows, csr, x, y_serial);
        end_time = omp_get_wtime();
        times[i] = end_time - start_time;
    }

    calculate_performance_openmp(times, mat, matrix_name, "serial", 1, time_serial, filename_p);

    // Reset the times array
    memset(times, 0, sizeof(times));

    // Loop through different thread numbers
    for (num_threads = 1; num_threads <= 40; num_threads++) {
        omp_set_num_threads(num_threads);
        printf("Number of threads: %d\n", num_threads);

        int *row_bounds = malloc((num_threads + 1) * sizeof(int));
        if (!row_bounds) {
            fprintf(stderr, "Allocation error for row_bounds.\n");
            exit(EXIT_FAILURE);
        }

        compute_row_bounds(csr, rows, num_threads, row_bounds);

        // Run the OpenMP CSR version and measure execution time
        for (i = 0; i < REPETITIONS; i++) {
            start_time = omp_get_wtime();
            prod_openmp_csr(rows, csr, x, y_csr, row_bounds);
            end_time = omp_get_wtime();
            times[i] = end_time - start_time;
        }

        // Verify correctness of CSR OpenMP result
        if (check_results(y_serial, y_csr, rows) == 0) {
            printf("Serial result is different from parallel result with csr\n");
        } else {
            printf("CSR results checked\n");
        }

        calculate_performance_openmp(times, mat, matrix_name, "CSR", omp_get_max_threads(), time_csr, filename_p);

        // Reset the times array
        memset(times, 0, sizeof(times));

        int *block_bounds = malloc((num_threads + 1) * sizeof(int));
        if (block_bounds == NULL) {
            fprintf(stderr, "Allocation error for row_bounds.\n");
            exit(EXIT_FAILURE);
        }
    
        compute_block_bounds(hll->numBlocks, num_threads, block_bounds);

        // Run the OpenMP HLL version and measure execution time
        for (i = 0; i < REPETITIONS; i++) {
            start_time = omp_get_wtime();
            prod_openmp_hll(hll, x, y_hll, block_bounds);
            end_time = omp_get_wtime();
            times[i] = end_time - start_time;
        }

        // Verify correctness of HLL OpenMP result
        if (check_results(y_serial, y_hll, rows) == 0) {
            printf("Serial result is different from parallel result with hll\n");
        } else {
            printf("HLL results checked\n");
        }

        calculate_performance_openmp(times, mat, matrix_name, "HLL", omp_get_max_threads(), time_hll, filename_p);

        // Compute and save speedup results
        calculate_speedup(matrix_name, *time_serial, *time_csr, *time_hll, filename_s, omp_get_max_threads(), nz);

        free(block_bounds);
        free(row_bounds);
    }

    // Free all allocated memory
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
        run_single_execution(matrix_name, mat);
    } else {
        run_all_threads_execution(matrix_name, mat);
    }

    return 0;
}
