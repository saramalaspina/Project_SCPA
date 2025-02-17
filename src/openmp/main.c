#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }

    MatrixElement *mat = read_matrix(argv[1]);
    if (!mat) exit(1);

    // Creazione del mediatore
    MatrixConversionMediator mediator = createMatrixMediator();

    // Creazione struct formato CSR
    CSRMatrix csr;

    // Dichiarazione struct formato HLL
    HLLMatrix hll;

    mediator.convertToCSR(mat->matrix, mat->nz, mat->M, &csr);
    mediator.convertToHLL(mat->matrix, mat->nz, mat->M, mat->N, &hll);

    // Allocazione dinamica dell'array
    double *x = malloc(mat->N * sizeof(double));
    if (x == NULL) {
        printf("Errore nell'allocazione di memoria.\n");
        return 1;
    }

    for (int i = 0; i < mat->N; i++) {
        x[i] = i + 1;
    }

    double *res = spmv_csr(mat->M, &csr, x);
    printf("Risultato calcolo seriale CSR:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res[i]);
    }

    double *res_csr = spmv_csr_parallel(mat->M, &csr, x);
    printf("Risultato calcolo parallelo CSR:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res_csr[i]);
    }

    double *res_hll = spmv_hll_parallel(&hll, x);
    printf("Risultato calcolo parallelo HLL:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res_hll[i]);
    }

    freeCSRMatrix(&csr);
    freeHLLMatrix(&hll);
    free(mat->matrix);
    free(mat);
    free(res);
    free(res_csr);
    free(res_hll);
    free(x);

    return 0;
}