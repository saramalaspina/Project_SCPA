#include <stdio.h>
#include <stdlib.h>
#include "../../lib/mmio.h"
#include "../../lib/utils.h"

// Reads a sparse matrix from a Matrix Market (.mtx) file
MatrixElement* read_matrix(char *file_name) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz; 
    int i, *I, *J;
    double *val;

    // Open the file for reading
    if ((f = fopen(file_name, "r")) == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }

    // Read the Matrix Market header
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    // Read the size of the matrix and the number of non-zero elements
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
        exit(1);

    // Allocate arrays for the COO format
    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));
    if (I == NULL || J == NULL || val == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    // Read the entries of the matrix
    // If it's a "pattern" matrix, read only row and column indices, and set value to 1.0
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
        // Convert from 1-based indexing to 0-based
        I[i]--;
        J[i]--;
    }
    fclose(f);

    // If the matrix is symmetric, add the symmetric counterpart for each off-diagonal element
    if (mm_is_symmetric(matcode)) {
        printf("Symmetric matrix\n");
        int additional = 0;
        for (i = 0; i < nz; i++) {
            if (I[i] != J[i])
                additional++;
        }
        int nz_tot = nz + additional;

        // Allocate new arrays to hold the symmetric matrix
        int *I_sym = (int *) malloc(nz_tot * sizeof(int));
        int *J_sym = (int *) malloc(nz_tot * sizeof(int));
        double *val_sym = (double *) malloc(nz_tot * sizeof(double));
        if (I_sym == NULL || J_sym == NULL || val_sym == NULL) {
            printf("Memory allocation failed\n");
            exit(1);
        }

        int pos = 0;
        for (i = 0; i < nz; i++) {
            I_sym[pos] = I[i];
            J_sym[pos] = J[i];
            val_sym[pos] = val[i];
            pos++;
            // If it's off-diagonal, add the symmetric counterpart
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

    // Allocate the COO element structure 
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

    // Allocate the final matrix 
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
