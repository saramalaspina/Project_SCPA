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

void calculate_performance_openmp(double *times, MatrixElement *mat, char *matrix_name, char *type, int numThreads, double *time, char* filename){
    double total_time = 0.0;

    for (int i = 10; i < REPETITIONS; i++){
        total_time += times[i];
    }

    total_time /= (REPETITIONS - 10); 
    double time_ms = total_time * 1000;

    double flops = (2.0 * mat->nz) / total_time;
    double gflops = flops / 1e9;

    printf("Risultati: gflops %.17g\n", gflops);

    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(1);
    }

    if (file_is_empty(fp)) {
        fprintf(fp, "matrix, M, N, nz, type, avgTime, avgGFlops, nThreads\n");
    }
    fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f, %d\n", matrix_name, mat->M, mat->N, mat->nz, type, time_ms, gflops, numThreads);

    fclose(fp);    

    *time = time_ms;

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

void calculate_performance_cuda(double *times, MatrixElement *mat, const char *matrix_name, const char *type, double *time){
    double total_time = 0.0;

    for (int i = 10; i < REPETITIONS; i++){
        total_time += times[i];
    }

    total_time /= (REPETITIONS - 10); 

    double flops = (2.0 * mat->nz) / (total_time/1000);
    double gflops = flops / 1e9;

    printf("Risultati: gflops %.17g\n", gflops);

    FILE *fp = fopen("results/cuda/performance.csv", "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(1);
    }
  
    if (file_is_empty(fp)) {
        fprintf(fp, "matrix, M, N, nz, type, avgTime, avgGFlops\n");
    }
    fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f\n", matrix_name, mat->M, mat->N, mat->nz, type, total_time, gflops);

    fclose(fp);    

    *time = total_time;

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



void calculate_speedup(const char* matrix_name, double time_serial, double time_csr, double time_hll, const char* file, int numThreads, int nz){

    double speedup_csr = time_serial/time_csr;
    double speedup_hll = time_serial/time_hll;

    printf("Speedup csr %.17g\n", speedup_csr);
    printf("Speedup hll %.17g\n", speedup_hll);

    FILE *fp = fopen(file, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        return;
    }

    if(numThreads != 0){
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, nz, time_serial, speedup_csr, speedup_hll, nThreads\n");
        }
        fprintf(fp, "%s, %d, %.6f, %.6f, %.6f, %d\n", matrix_name, nz, time_serial, speedup_csr, speedup_hll, numThreads);
    } else {
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, nz, time_serial, speedup_csr, speedup_hll\n");
        }
        fprintf(fp, "%s, %d, %.6f, %.6f, %.6f\n", matrix_name, nz, time_serial, speedup_csr, speedup_hll);
    }
   
    fclose(fp);    
    
}

int compare_coo(const void *a, const void *b) {
    COOElement *elem1 = (COOElement *)a;
    COOElement *elem2 = (COOElement *)b;

    if (elem1->row != elem2->row)
        return elem1->row - elem2->row; // Ordina per riga crescente
    return elem1->col - elem2->col;     // Se le righe sono uguali, ordina per colonna crescente
}
