#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void freeHLLMatrixCuda(HLLMatrix *hll) {
    for (int b = 0; b < hll->num_blocks; b++) {
        for (int i = 0; i < hll->blocks[b].rows; i++) {
            free(hll->blocks[b].JA[i]);
            free(hll->blocks[b].AS[i]);
        }
        free(hll->blocks[b].JA);
        free(hll->blocks[b].AS);
        free(hll->blocks[b].JA_t);
        free(hll->blocks[b].AS_t);
    }
    free(hll->blocks);
}

void freeHLLMatrixOpenmp(HLLMatrix *hll) {
    for (int b = 0; b < hll->num_blocks; b++) {
        for (int i = 0; i < hll->blocks[b].rows; i++) {
            free(hll->blocks[b].JA[i]);
            free(hll->blocks[b].AS[i]);
        }
        free(hll->blocks[b].JA);
        free(hll->blocks[b].AS);
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
