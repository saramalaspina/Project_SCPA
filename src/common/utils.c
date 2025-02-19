#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

void freeHLLMatrix(HLLMatrix *hll) {
    for (int b = 0; b < hll->num_blocks; b++) {
        for (int i = 0; i < hll->blocks[b].rows; i++) {
            free(hll->blocks[b].JA[i]);
            free(hll->blocks[b].AS[i]);
        }
        free(hll->blocks[b].JA);
        free(hll->blocks[b].AS);
        if(hll->blocks[b].JA_t != NULL && hll->blocks[b].AS_t != NULL){
            free(hll->blocks[b].JA_t);
            free(hll->blocks[b].AS_t);
        }
    }
    free(hll->blocks);
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

void printELLBlockTransposed(const ELLBlock *block) {
    printf("JA_t (trasposta) =\n");
    for (int r = 0; r < block->max_nz; r++) {
        for (int c = 0; c < block->rows; c++) {
            int idx = r * block->rows + c;
            printf(" %d", block->JA_t[idx]);
        }
        printf("\n");
    }
    printf("AS_t (trasposta) =\n");
    for (int r = 0; r < block->max_nz; r++) {
        for (int c = 0; c < block->rows; c++) {
            int idx = r * block->rows + c;
            printf(" %.1f", block->AS_t[idx]);
        }
        printf("\n");
    }
}


void printHLLMatrixTransposed(const HLLMatrix *H) {
    printf("Transposed HLL Matrix with %d block(s)\n", H->num_blocks);
    for (int b = 0; b < H->num_blocks; b++) {
        printf("Block %d:\n", b);
        printELLBlockTransposed(&H->blocks[b]);
    }
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

void calculatePerformance(double *times, MatrixElement *mat, char *matrix_name, char *type, char *paral){
    double total_time = 0.0;
    char file[256];
    sprintf(file, "results/%s/performance.csv", paral);

    for (int i = 1; i < REPETITIONS; i++){
        total_time += times[i];
    }

    total_time /= (REPETITIONS - 1); 
    double time_ms = total_time * 1000;

    double flops = (2.0 * mat->nz) / total_time;
    double gflops = flops / 1e9;

    printf("Risultati: gflops %.17g\n", gflops);

    FILE *fp = fopen(file, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        return;
    }

    if(strcmp(paral, "openmp")==0){
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, M, N, nz, type, avgTime, avgFlops, avgGFlops, nThreads\n");
        }
        fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f, %.6f, %d\n",matrix_name, mat->M, mat->N, mat->nz, type, time_ms, flops, gflops, omp_get_max_threads());
    } else {
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, M, N, nz, type, avgTime, avgFlops, avgGFlops\n");
        }
        fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f, %.6f\n", matrix_name, mat->M, mat->N, mat->nz, type, time_ms, flops, gflops);
    }
    fclose(fp);    

}

int checkResults(double* arr1, double* arr2, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(arr1[i] - arr2[i]) > 1e-9) {
            return 0; // Gli array sono diversi
        }
    }
    return 1; // Gli array sono uguali
}
