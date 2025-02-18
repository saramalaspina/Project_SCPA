#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Definizione COO->CSR Converter
typedef struct {
    void (*convert)(COOElement *coo, int nz, int m);
} COOtoCSRConverter;

void convertCOOtoCSR(COOElement *coo, int nz, int m, CSRMatrix *matrix) {
    printf("Converting COO to CSR...\n");

    matrix->IRP = (int *)malloc((m + 1) * sizeof(int));
    matrix->JA = (int *)malloc(nz * sizeof(int));
    matrix->AS = (double *)malloc(nz * sizeof(double));

    // Inizializzazione degli array CSR
    for (int i = 0; i <= m; i++) {
        matrix->IRP[i] = 0;
    }

    // Costruzione di row_ptr (conteggio elementi per riga)
    for (int i = 0; i < nz; i++) {
        matrix->IRP[coo[i].row + 1]++;
    }

    // Conversione dei conteggi in indici cumulativi
    for (int i = 1; i <= m; i++) {
        matrix->IRP[i] += matrix->IRP[i - 1];
    }

    // Array temporaneo per tracciare le posizioni di inserimento
    int *row_counter = (int *)malloc(m * sizeof(int));
    memcpy(row_counter, matrix->IRP, m * sizeof(int));

    // Riempie JA e AS rispettando l'ordine corretto delle righe
    for (int i = 0; i < nz; i++) {
        int row = coo[i].row;
        int index = row_counter[row];  // Posizione corretta
        matrix->JA[index] = coo[i].col;
        matrix->AS[index] = coo[i].value;
        row_counter[row]++;  // Avanza la posizione per la prossima scrittura
    }

    free(row_counter);  // Rilascia la memoria allocata

   /* // Stampa del formato CSR
    printf("IRP: ");
    for (int i = 0; i <= m; i++) {
        printf("%d ", matrix->IRP[i]);
    }
    printf("\n");

    printf("JA: ");
    for (int i = 0; i < nz; i++) {
        printf("%d ", matrix->JA[i]);
    }
    printf("\n");

    printf("AS: ");
    for (int i = 0; i < nz; i++) {
        printf("%20.19g ", matrix->AS[i]);
    }
    printf("\n");*/

}

// Definizione COO->HLL Converter
typedef struct {
    void (*convert)(COOElement *coo, int nz, int m);
} COOtoHLLConverter;


void convertCOOtoHLL(COOElement *coo, int nz, int M, int N, HLLMatrix *hll) {
    printf("Converting COO to HLL...\n");
    int num_blocks = (M + HACKSIZE - 1) / HACKSIZE;
    hll->num_blocks = num_blocks;
    hll->blocks = (ELLBlock *)malloc(num_blocks * sizeof(ELLBlock));

    for (int b = 0; b < num_blocks; b++) {
        int start_row = b * HACKSIZE;
        int end_row = (start_row + HACKSIZE < M) ? (start_row + HACKSIZE) : M;
        int block_rows = end_row - start_row;

        int *row_nnz = (int *)calloc(block_rows, sizeof(int));
        for (int i = 0; i < nz; i++) {
            if (coo[i].row >= start_row && coo[i].row < end_row)
                row_nnz[coo[i].row - start_row]++;
        }

        int max_nz = 0;
        for (int i = 0; i < block_rows; i++) {
            if (row_nnz[i] > max_nz) max_nz = row_nnz[i];
        }

        hll->blocks[b].rows = block_rows;
        hll->blocks[b].cols = N;
        hll->blocks[b].max_nz = max_nz;
        hll->blocks[b].JA = (int **)malloc(block_rows * sizeof(int *));
        hll->blocks[b].AS = (double **)malloc(block_rows * sizeof(double *));

        for (int i = 0; i < block_rows; i++) {
            hll->blocks[b].JA[i] = (int *)malloc(max_nz * sizeof(int));
            hll->blocks[b].AS[i] = (double *)malloc(max_nz * sizeof(double));
            memset(hll->blocks[b].JA[i], -1, max_nz * sizeof(int));
            memset(hll->blocks[b].AS[i], 0, max_nz * sizeof(double));
        }

        int *current_nz = (int *)calloc(block_rows, sizeof(int));
        for (int i = 0; i < nz; i++) {
            int row = coo[i].row - start_row;
            if (row >= 0 && row < block_rows) {
                int pos = current_nz[row]++;
                hll->blocks[b].JA[row][pos] = coo[i].col;
                hll->blocks[b].AS[row][pos] = coo[i].value;
            }
        }

        for (int i = 0; i < block_rows; i++) {
            if (current_nz[i] > 0) {
                for (int j = current_nz[i]; j < max_nz; j++) {
                    hll->blocks[b].JA[i][j] = hll->blocks[b].JA[i][j - 1];
                }
            }
        }        

        /*printf("HLL Matrix with %d blocks\n", hll -> num_blocks);
        for (int b = 0; b < hll -> num_blocks; b++) {
            printf("Block %d:\n", b);
            printf("JA =\n");
            for (int i = 0; i < hll -> blocks[b].rows; i++) {
                printf("  ");
                for (int j = 0; j < hll -> blocks[b].max_nz; j++) {
                    printf("%d ", hll -> blocks[b].JA[i][j]);
                }
                printf("\n");
            }
            printf("AS =\n");
            for (int i = 0; i < hll -> blocks[b].rows; i++) {
                printf("  ");
                for (int j = 0; j < hll -> blocks[b].max_nz; j++) {
                    printf("%.1f ", hll -> blocks[b].AS[i][j]);
                }
                printf("\n");
            }
        }*/

        free(row_nnz);
        free(current_nz);
    }
}


void transposeELLBlock(ELLBlock *block) {
    int rows = block->rows;
    int max_nz = block->max_nz;
    
    // Alloca gli array contigui per la versione trasposta
    block->JA_t = malloc(rows * max_nz * sizeof(int));
    block->AS_t = malloc(rows * max_nz * sizeof(double));
    
    // Esegui la trasposizione: per ogni elemento (r, c) nell'originale
    // copia in posizione (c, r) nel vettore trasposto
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < max_nz; c++) {
            block->JA_t[c * rows + r] = block->JA[r][c];
            block->AS_t[c * rows + r] = block->AS[r][c];
        }
    }
}

void trasponseHLLMatrix(HLLMatrix *hll) {
    for (int b = 0; b < hll->num_blocks; b++) {
        ELLBlock *block = &hll->blocks[b];
        transposeELLBlock(block);
    }
}


