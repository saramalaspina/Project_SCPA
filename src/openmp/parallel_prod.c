#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>

void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y) {
    int chunk_size = 8; // Imposta una dimensione di chunk ragionevole

    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < M; i++) {  // Scorre le righe della matrice
        double sum = 0.0;  // Variabile privata per evitare race conditions

        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {  // Scorre elementi non nulli della riga i
            sum += csr->AS[j] * x[csr->JA[j]];
        }

        y[i] = sum;
    }
}


void prodOpenmpHLL(HLLMatrix *hll, double *x, double *y) {
    int total_rows = hll->num_blocks * HACKSIZE;  // Numero totale di righe

    int chunk_size = 2;  // Regola per bilanciare il carico

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


