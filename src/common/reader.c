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

    if ((f = fopen(file_name, "r")) == NULL)
    {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));
    J = (int *) malloc(nz * sizeof(int));
    val = (double *) malloc(nz * sizeof(double));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }

    if (f !=stdin) fclose(f);

    /************************/
    /* now write out matrix */
    /************************/

    mm_write_banner(stdout, matcode);
    mm_write_mtx_crd_size(stdout, M, N, nz);
    for (i=0; i<nz; i++)
        fprintf(stdout, "%d %d %20.19g\n", I[i], J[i], val[i]);

    COOElement *coo = malloc(nz * sizeof(COOElement));
    if (coo == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (i=0; i<nz; i++) {
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
