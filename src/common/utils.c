#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void freeHLLMatrix(HLLMatrix *hll) {
    /* Liberazione della memoria allocata */
    for (int b = 0; b < hll->numBlocks; b++) {
        free(hll->blocks[b].JA);
        free(hll->blocks[b].AS);
    }
    free(hll->blocks);
    free(hll);
}

double *generateVector(int N){
    // Allocazione dinamica del vettore x
    double *x = (double *)malloc(N * sizeof(double));
    if (!x) {
        fprintf(stderr, "Errore nell'allocazione di memoria per x.\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
    }
    return x;
}


void freeCSRMatrix(CSRMatrix *csr){
    free(csr->IRP);
    free(csr->JA);
    free(csr->AS);
}

void printResult(double *y, int M){
    for(int i=0; i<M; i++){
        printf("%lf\n", y[i]);
    }
}

int file_is_empty(FILE *fp) {
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    return size == 0;
}

void calculatePerformanceOpenMP(double *times, MatrixElement *mat, char *matrix_name, char *type, int numThreads, double *time){
    double total_time = 0.0;

    for (int i = 1; i < REPETITIONS; i++){
        total_time += times[i];
    }

    total_time /= (REPETITIONS - 1); 
    double time_ms = total_time * 1000;

    double flops = (2.0 * mat->nz) / total_time;
    double gflops = flops / 1e9;

    printf("Risultati: gflops %.17g\n", gflops);

    FILE *fp = fopen("results/openmp/performance.csv", "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(1);
    }

    if (file_is_empty(fp)) {
        fprintf(fp, "matrix, M, N, nz, type, avgTime, avgGFlops, nThreads\n");
    }
    fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f, %d\n",matrix_name, mat->M, mat->N, mat->nz, type, time_ms, gflops, numThreads);

    fclose(fp);    

    *time = time_ms;

}

void calculatePerformanceCuda(double *times, MatrixElement *mat, char *matrix_name, char *type, double *time){
    double total_time = 0.0;

    for (int i = 1; i < REPETITIONS; i++){
        total_time += times[i];
    }

    total_time /= (REPETITIONS - 1); 

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


int checkResults(double* arr1, double* arr2, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(arr1[i] - arr2[i]) > 1e-6) {
            return 0; // Gli array sono diversi
        }
    }
    return 1; // Gli array sono uguali
}

void calculateSpeedup(char *matrix_name, double time_serial, double time_csr, double time_hll, char *paral, int numThreads){
    double total_time = 0.0;
    char file[256];
    sprintf(file, "results/%s/speedup.csv", paral);

    double speedup_csr = time_serial/time_csr;
    double speedup_hll = time_serial/time_hll;

    printf("Speedup csr %.17g\n", speedup_csr);
    printf("Speedup hll %.17g\n", speedup_hll);

    FILE *fp = fopen(file, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        return;
    }

    if(strcmp(paral, "openmp")==0){
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, time_serial, speedup_csr, speedup_hll, nThreads\n");
        }
        fprintf(fp, "%s, %.6f, %.6f, %.6f, %d\n", matrix_name, time_serial, speedup_csr, speedup_hll, numThreads);
    } else {
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, time_serial, speedup_csr, speedup_hll\n");
        }
        fprintf(fp, "%s, %.6f, %.6f, %.6f\n", matrix_name, time_serial, speedup_csr, speedup_hll);
    }
   
    fclose(fp);    
    
}

int compareCOO(const void *a, const void *b) {
    COOElement *elem1 = (COOElement *)a;
    COOElement *elem2 = (COOElement *)b;

    if (elem1->row != elem2->row)
        return elem1->row - elem2->row; // Ordina per riga crescente
    return elem1->col - elem2->col;     // Se le righe sono uguali, ordina per colonna crescente
}
