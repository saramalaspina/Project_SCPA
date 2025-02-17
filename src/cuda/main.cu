#include <stdio.h>
#include <stdlib.h>
#include "../../lib/utils.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
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

    trasponseHLLMatrix(&hll);
    printHLLMatrixTransposed(&hll);

    // Allocazione dinamica del vettore x
    double *x = (double *)malloc(mat->N * sizeof(double));
    if (!x) {
        fprintf(stderr, "Errore nell'allocazione di memoria per x.\n");
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

    // **Esecuzione su GPU (CSR)**
    double *res_csr_cuda = spmv_csr_cuda(mat->M, mat->N, csr.IRP, csr.JA, csr.AS, x);
    printf("Risultato calcolo CUDA CSR:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res_csr_cuda[i]);
    }

    double *res_hll_cuda = spmv_hll_cuda(&hll, mat->M, mat->N, x);
    printf("Risultato calcolo CUDA HLL:\n");
    for (int i = 0; i < mat->M; i++) {
        printf("%lg\n", res_hll_cuda[i]);
    }

    free(x);
    freeHLLMatrix(&hll);
    freeCSRMatrix(&csr);
    free(mat->matrix);
    free(mat);
    free(res);
    free(res_csr_cuda);
    free(res_hll_cuda);

    return 0;
}
