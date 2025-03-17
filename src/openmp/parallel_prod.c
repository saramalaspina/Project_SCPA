#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>

void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y) {
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
}


void prodOpenmpHLL(HLLMatrix *hll, double *x, double *y) {
    int chunk_size = (hll->num_blocks > 500) ? 8 : (hll->num_blocks > 100) ? 4 : 2;

    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int b = 0; b < hll->num_blocks; b++) {  // Per ogni blocco
        ELLBlock *block = &hll->blocks[b];

        for (int i = 0; i < block->rows; i++) {  // Per ogni riga del blocco
            double sum = 0.0;  // Variabile privata per ogni thread

            for (int j = 0; j < block->max_nz; j++) {  // Per ogni elemento non nullo
                sum += block->AS[i][j] * x[block->JA[i][j]];
            }

            y[b * HACKSIZE + i] = sum;  // Scrittura senza conflitti
        }
    }
}


