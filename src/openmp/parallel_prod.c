#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y, int *row_bounds) {
    int num_threads = omp_get_max_threads();

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        for (int i = row_bounds[tid]; i < row_bounds[tid+1]; i++) {
            double sum = 0.0;
            int start = csr->IRP[i];
            int end   = csr->IRP[i+1];

            for (int j = start; j < end; j++) {
                sum += csr->AS[j] * x[csr->JA[j]];
            }
            y[i] = sum;
        }
    }
}


void prodOpenmpHLL(HLLMatrix *hll, double *x, double *y) {

    /* Parallelizzazione sul ciclo esterno (sui blocchi) */
    #pragma omp parallel for schedule(guided)
    for (int b = 0; b < hll->numBlocks; b++) {
        EllpackBlock *block = &hll->blocks[b];
        /* Il blocco b inizia dalla riga globale base = b * hackSize */
        int base = b * hll->hackSize;
        for (int i = 0; i < block->block_rows; i++) {
            int global_row = base + i;
            double sum = 0.0;
            for (int j = 0; j < block->maxnz; j++) {
                int idx = i * block->maxnz + j;
                int col = block->JA[idx];
                if (col != -1) {  // controllo che l'elemento sia valido
                    sum += block->AS[idx] * x[col];
                }
            }
            y[global_row] = sum;
        }
    }
}

    


