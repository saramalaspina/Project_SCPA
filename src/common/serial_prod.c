#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>

double *spmv_csr(int M, CSRMatrix *csr, double *x) {

    double *y = calloc(M, sizeof(double));  // Allocazione del risultato
    if (!y) {
        fprintf(stderr, "Errore di allocazione della memoria\n");
        exit(1);
    }

    for (int i = 0; i < M; i++) {  // Scorre le righe della matrice
        double sum = 0.0;

        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {  // Scorre gli elementi non nulli della riga i
            sum += csr->AS[j] * x[csr->JA[j]];
        }

        y[i] = sum;
    }

    return y;
}





