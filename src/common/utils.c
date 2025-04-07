#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

void free_hll_matrix(HLLMatrix *hll) {
    /* Liberazione della memoria allocata */
    for (int b = 0; b < hll->numBlocks; b++) {
        free(hll->blocks[b].JA);
        free(hll->blocks[b].AS);
    }
    free(hll->blocks);
    free(hll);
}

double *generate_vector(const char *matrix_name, int N) {
    // Imposta il seed per la riproducibilità
    srand(1234);

    // Allocazione dinamica del vettore x
    double *x = (double *)malloc(N * sizeof(double));
    if (!x) {
        fprintf(stderr, "Errore nell'allocazione di memoria per x.\n");
        exit(1);
    }

    // Popolamento con valori casuali tra 0.1 e 2.0
    for (int i = 0; i < N; i++) {
        x[i] = 0.1 + ((double)rand() / RAND_MAX) * (2.0 - 0.1);
    }
    
    return x;
}


void free_csr_matrix(CSRMatrix *csr){
    free(csr->IRP);
    free(csr->JA);
    free(csr->AS);
}

void print_result(double *y, int M){
    for(int i=0; i<M; i++){
        printf("%lf\n", y[i]);
    }
}

int file_is_empty(FILE *fp) {
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    return size == 0;
}

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

void compute_row_bounds(CSRMatrix *csr, int M, int num_threads, int *row_bounds) {
    int total_nnz = csr->IRP[M];
    int target_nnz_per_thread = (total_nnz + num_threads - 1) / num_threads;

    row_bounds[0] = 0;
    // Per ogni thread, determina il confine di riga in base al target cumulativo
    for (int t = 1; t < num_threads; t++) {
        int target = t * target_nnz_per_thread;
        row_bounds[t] = binary_search_csr(csr->IRP, M + 1, target);
    }
    row_bounds[num_threads] = M;
}

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
            printf("Mismatch at index %d: seriale = %f, parallelo = %f (rel_diff=%lf)\n", i, y_serial[i], y_parallel[i], rel_diff_val);
        }
    }

    return passed;
}


int compare_coo(const void *a, const void *b) {
    COOElement *elem1 = (COOElement *)a;
    COOElement *elem2 = (COOElement *)b;

    if (elem1->row != elem2->row)
        return elem1->row - elem2->row; // Ordina per riga crescente
    return elem1->col - elem2->col;     // Se le righe sono uguali, ordina per colonna crescente
}
