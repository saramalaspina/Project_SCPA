#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>

void prod_serial(int M, CSRMatrix *csr, double *x, double *y) {

    for (int i = 0; i < M; i++) {  // Scorre le righe della matrice
        double sum = 0.0;

        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {  // Scorre gli elementi non nulli della riga i

            sum += csr->AS[j] * x[csr->JA[j]];
        }
        y[i] = sum;
    }
}





