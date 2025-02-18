#include <stdio.h>
#include <stdlib.h>
#include "../../lib/mmio.h"
#include "../../lib/utils.h"

MatrixElement* read_matrix(char *file_name) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int i, *I, *J;
    double *val;

    if ((f = fopen(file_name, "r")) == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /* Trova le dimensioni della matrice sparsa */
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        exit(1);

    /* Allocazione degli array per la lettura */
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
    if (I == NULL || J == NULL || val == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    /* Lettura degli elementi dal file:
       - Se la matrice è di tipo "pattern" vengono letti solo I e J,
         e il valore viene impostato a 1.0.
       - Altrimenti, viene letto anche il valore esplicito.
    */
    for (i = 0; i < nz; i++) {
        if (mm_is_pattern(matcode)) {
            if (fscanf(f, "%d %d\n", &I[i], &J[i]) != 2) {
                printf("Error reading pattern matrix entry\n");
                exit(1);
            }
            val[i] = 1.0;
        } else {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]) != 3) {
                printf("Error reading matrix entry\n");
                exit(1);
            }
        }
        /* conversione da 1-based a 0-based */
        I[i]--;
        J[i]--;
    }
    fclose(f);

    /* Gestione delle matrici simmetriche:
       Se la matrice è simmetrica, per ogni elemento fuori diagonale,
       aggiungiamo la sua controparte speculare.
    */
    if (mm_is_symmetric(matcode)) {
        printf("Symmetric matrix\n");
        int additional = 0;
        for (i = 0; i < nz; i++) {
            if (I[i] != J[i])
                additional++;
        }
        int nz_tot = nz + additional;
        int *I_sym = (int *) malloc(nz_tot * sizeof(int));
        int *J_sym = (int *) malloc(nz_tot * sizeof(int));
        double *val_sym = (double *) malloc(nz_tot * sizeof(double));
        if (I_sym == NULL || J_sym == NULL || val_sym == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }
        int pos = 0;
        for (i = 0; i < nz; i++) {
            /* Copia l'elemento originale */
            I_sym[pos] = I[i];
            J_sym[pos] = J[i];
            val_sym[pos] = val[i];
            pos++;
            /* Se l'elemento è fuori diagonale, aggiungi anche la controparte */
            if (I[i] != J[i]) {
                I_sym[pos] = J[i];
                J_sym[pos] = I[i];
                val_sym[pos] = val[i];
                pos++;
            }
        }
        nz = nz_tot;
        free(I);
        free(J);
        free(val);
        I = I_sym;
        J = J_sym;
        val = val_sym;
    }

   
    /*mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i = 0; i < nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i], J[i], val[i]);*/

    /* Trasferimento dei dati in una struttura COO (o altra struttura scelta) */
    COOElement *coo = malloc(nz * sizeof(COOElement));
    if (coo == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    for (i = 0; i < nz; i++) {
        coo[i].row = I[i];
        coo[i].col = J[i];
        coo[i].value = val[i];
    }

    MatrixElement *mat = malloc(sizeof(MatrixElement));
    if (mat == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }
    mat->M = M;
    mat->N = N;
    mat->nz = nz;
    mat->matrix = coo;

    free(I);
    free(J);
    free(val);

    return mat;
}
