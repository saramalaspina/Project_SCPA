#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// Frees memory allocated for a HLL matrix
void free_hll_matrix(HLLMatrix *hll) {
    for (int b = 0; b < hll->numBlocks; b++) {
        free(hll->blocks[b].JA);  
        free(hll->blocks[b].AS);  
    }
    free(hll->blocks);  
    free(hll);          
}

// Frees memory allocated for a CSR matrix
void free_csr_matrix(CSRMatrix *csr){
    free(csr->IRP);  
    free(csr->JA);  
    free(csr->AS);  
}

// Generates a random vector of length N with values between 0.1 and 2.0
double *generate_vector(int N) {
    srand(1234);  // Set seed for reproducibility

    double *x = (double *)malloc(N * sizeof(double));
    if (!x) {
        fprintf(stderr, "Error allocating memory for x.\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        x[i] = 0.1 + ((double)rand() / RAND_MAX) * (2.0 - 0.1);
    }

    return x;
}

void print_result(double *y, int M){
    for(int i = 0; i < M; i++){
        printf("%lf\n", y[i]);
    }
}

// Performs a binary search on the IRP array to find the row containing a given target non-zero index
int binary_search_csr(const int *IRP, int size, int target) {
    int low = 0;
    int high = size;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (IRP[mid] <= target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low - 1;
}

// Computes row boundaries for dividing CSR matrix rows among threads, trying to balance non-zeros between threads
void compute_row_bounds(CSRMatrix *csr, int M, int num_threads, int *row_bounds) {
    int total_nnz = csr->IRP[M];  
    int target_nnz_per_thread = (total_nnz + num_threads - 1) / num_threads;

    row_bounds[0] = 0;
    for (int t = 1; t < num_threads; t++) {
        int target = t * target_nnz_per_thread;
        row_bounds[t] = binary_search_csr(csr->IRP, M + 1, target);
    }
    row_bounds[num_threads] = M;
}

// Compares the results of serial and parallel computations
int check_results(double *y_serial, double *y_parallel, int size) {
    double diff = 0.0;
    double rel_diff = 0.0;
    int passed = 1;

    for (int i = 0; i < size; i++) {
        double abs_diff = fabs(y_serial[i] - y_parallel[i]);
        double max_val = fmax(fabs(y_serial[i]), fabs(y_parallel[i]));
        double rel_diff_val = (max_val == 0) ? 0.0 : abs_diff / max_val;

        diff = fmax(diff, abs_diff);
        rel_diff = fmax(rel_diff, rel_diff_val);

        if (abs_diff > 1e-9 && rel_diff_val > 1e-6) {
            passed = 0;
            printf("Mismatch at index %d: serial = %f, parallel = %f (rel_diff=%lf)\n",
                   i, y_serial[i], y_parallel[i], rel_diff_val);
        }
    }

    return passed;
}

// Comparison function for sorting COO elements first by row, then by column
int compare_coo(const void *a, const void *b) {
    COOElement *elem1 = (COOElement *)a;
    COOElement *elem2 = (COOElement *)b;

    if (elem1->row != elem2->row)
        return elem1->row - elem2->row;  // Sort by row
    return elem1->col - elem2->col;      // If same row, sort by column
}
