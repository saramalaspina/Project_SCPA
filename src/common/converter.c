#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CSRMatrix *convert_coo_to_csr(COOElement *coo, int nz, int m) {
    printf("Converting COO to CSR...\n");
    
    // Allocate the CSR matrix structure
    CSRMatrix *matrix = (CSRMatrix *) malloc(sizeof(CSRMatrix));
    if (!matrix) {
        fprintf(stderr, "Errore di allocazione per CSRMatrix\n");
        exit(EXIT_FAILURE);
    }

    matrix->IRP = (int *)malloc((m + 1) * sizeof(int));
    matrix->JA = (int *)malloc(nz * sizeof(int));
    matrix->AS = (double *)malloc(nz * sizeof(double));

    // Initialize array
    for (int i = 0; i <= m; i++) {
        matrix->IRP[i] = 0;
    }

    // Count number of elements per row
    for (int i = 0; i < nz; i++) {
        matrix->IRP[coo[i].row + 1]++;
    }

    // Conversion into cumulative indexes
    for (int i = 1; i <= m; i++) {
        matrix->IRP[i] += matrix->IRP[i - 1];
    }

    // Temporary array to track insertion locations
    int *row_counter = (int *)malloc(m * sizeof(int));
    memcpy(row_counter, matrix->IRP, m * sizeof(int));

    // Fill JA and AS 
    for (int i = 0; i < nz; i++) {
        int row = coo[i].row;
        int index = row_counter[row]; 
        matrix->JA[index] = coo[i].col;
        matrix->AS[index] = coo[i].value;
        row_counter[row]++;  
    }

    free(row_counter);  

    return matrix;

}


HLLMatrix *convert_coo_to_hll(MatrixElement *coo, int hackSize) {
    printf("Converting COO to HLL...\n");

    // Validate input
    if (!coo || coo->nz < 0 || hackSize <= 0)
        return NULL;

    // Calculate number of blocks based on hackSize
    int numBlocks = (coo->M + hackSize - 1) / hackSize;

    // Allocate the HLL matrix structure
    HLLMatrix *hll = (HLLMatrix *) malloc(sizeof(HLLMatrix));
    if (!hll) {
        fprintf(stderr, "Memory allocation failed for HLLMatrix\n");
        exit(EXIT_FAILURE);
    }

    hll->hackSize = hackSize;
    hll->numBlocks = numBlocks;

    // Allocate memory for the ELLPACK blocks
    hll->blocks = (EllpackBlock *) malloc(numBlocks * sizeof(EllpackBlock));
    if (!hll->blocks) {
        fprintf(stderr, "Memory allocation failed for ELLPACK blocks\n");
        free(hll);
        exit(EXIT_FAILURE);
    }

    int coo_index = 0;  // index to iterate through COO matrix elements

    // Process each block
    for (int b = 0; b < numBlocks; b++) {
        int startRow = b * hackSize;
        int endRow = (b + 1) * hackSize;
        if (endRow > coo->M)
            endRow = coo->M;
        int blockRows = endRow - startRow;

        // Track non-zeros per row to determine the block max width
        int *nnz_per_row = (int *) calloc(blockRows, sizeof(int));
        if (!nnz_per_row) {
            fprintf(stderr, "Memory allocation failed for nnz_per_row\n");
            exit(EXIT_FAILURE);
        }

        int block_maxnz = 0;
        int temp_index = coo_index;

        // Count non-zeros for each row in the block
        for (int i = 0; i < blockRows; i++) {
            int currentRow = startRow + i;
            int count = 0;
            while (temp_index < coo->nz && coo->matrix[temp_index].row == currentRow) {
                count++;
                temp_index++;
            }
            nnz_per_row[i] = count;
            if (count > block_maxnz)
                block_maxnz = count;
        }

        // Allocate ELLPACK arrays for this block (column-major format)
        int *JA = (int *) malloc(blockRows * block_maxnz * sizeof(int));
        double *AS = (double *) malloc(blockRows * block_maxnz * sizeof(double));
        if (!JA || !AS) {
            fprintf(stderr, "Memory allocation failed for block arrays\n");
            exit(EXIT_FAILURE);
        }

        // Initialize arrays
        for (int i = 0; i < blockRows * block_maxnz; i++) {
            JA[i] = -1;
            AS[i] = 0.0;
        }

        // Fill JA and AS arrays with COO data
        for (int i = 0; i < blockRows; i++) {
            int currentRow = startRow + i;
            int count = 0;
            while (coo_index < coo->nz && coo->matrix[coo_index].row == currentRow) {
                if (count >= block_maxnz) {
                    fprintf(stderr, "Overflow: too many elements in row\n");
                    break;
                }
                // Use column-major order: count * blockRows + i
                int index = count * blockRows + i;
                JA[index] = coo->matrix[coo_index].col;
                AS[index] = coo->matrix[coo_index].value;
                count++;
                coo_index++;
            }
        }

        // Save the block's data
        hll->blocks[b].block_rows = blockRows;
        hll->blocks[b].N = coo->N;
        hll->blocks[b].maxnz = block_maxnz;
        hll->blocks[b].JA = JA;
        hll->blocks[b].AS = AS;

        free(nnz_per_row);  
    }

    return hll; 
}
