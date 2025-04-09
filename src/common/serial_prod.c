#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>

// Performs a serial matrix-vector multiplication: y = A * x
// A is stored in CSR format

void prod_serial(int M, CSRMatrix *csr, double *x, double *y) {

    for (int i = 0; i < M; i++) {  // Iterate over the rows of the matrix
        double sum = 0.0;

        // Iterate over the non-zero elements in row i
        for (int j = csr->IRP[i]; j < csr->IRP[i + 1]; j++) {
            // Multiply the matrix value by the corresponding element in vector x and accumulate the result
            sum += csr->AS[j] * x[csr->JA[j]];
        }

        y[i] = sum;
    }
}






