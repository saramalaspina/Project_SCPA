#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>

/*void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y) {
    int chunk_size = (M > 10000) ? 256 : 8;

    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < M; i++) {  // Scorre le righe della matrice
        double sum = 0.0;  // Variabile privata per evitare race conditions

        int start = csr->IRP[i];
        int end = csr->IRP[i + 1];

        for (int j = start; j < end; j++) {  // Scorre elementi non nulli della riga i
            sum += csr->AS[j] * x[csr->JA[j]];
        }

        y[i] = sum;
    }
}*/

/*void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y) {

    #pragma omp parallel for schedule(guided, 1)
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        int start = csr->IRP[i];
        int end = csr->IRP[i + 1];

        for (int j = start; j < end; j++) {
            sum += csr->AS[j] * x[csr->JA[j]];
        }
        y[i] = sum;
    }
}*/



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
    int chunk_size = (hll->numBlocks > 500) ? 8 : (hll->numBlocks > 100) ? 4 : 2;
    
    /* Parallelizzazione sul ciclo esterno (sui blocchi) */
    #pragma omp parallel for schedule(dynamic, chunk_size)
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

    


