#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CSRMatrix *convert_coo_to_csr(COOElement *coo, int nz, int m) {
    printf("Converting COO to CSR...\n");
    
    CSRMatrix *matrix = (CSRMatrix *) malloc(sizeof(CSRMatrix));
    if (!matrix) {
        fprintf(stderr, "Errore di allocazione per CSRMatrix\n");
        exit(EXIT_FAILURE);
    }

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

    return matrix;

}


/* Funzione di conversione da COO a HLL */
HLLMatrix *convert_coo_to_hll(MatrixElement *coo, int hackSize) {
    printf("Converting COO to HLL...\n");

    if (!coo || coo->nz < 0 || hackSize <= 0)
        return NULL;
    
    // Calcolo il numero di blocchi necessari
    int numBlocks = (coo->M + hackSize - 1) / hackSize;
    HLLMatrix *hll = (HLLMatrix *) malloc(sizeof(HLLMatrix));
    if (!hll) {
        fprintf(stderr, "Errore di allocazione per HLLMatrix\n");
        exit(EXIT_FAILURE);
    }
    hll->hackSize = hackSize;
    hll->numBlocks = numBlocks;
    hll->blocks = (EllpackBlock *) malloc(numBlocks * sizeof(EllpackBlock));
    if (!hll->blocks) {
        fprintf(stderr, "Errore di allocazione per i blocchi ELLPACK\n");
        free(hll);
        exit(EXIT_FAILURE);
    }
    
    /* coo_index tiene traccia della posizione corrente nell'array COO, 
       sfruttando il fatto che la matrice è ordinata per riga */
    int coo_index = 0;
    for (int b = 0; b < numBlocks; b++) {
        int startRow = b * hackSize;
        int endRow = (b + 1) * hackSize;
        if (endRow > coo->M)
            endRow = coo->M;
        int blockRows = endRow - startRow;
        
        /* Array temporaneo per memorizzare il numero di non-zeri per ogni riga del blocco */
        int *nnz_per_row = (int *) calloc(blockRows, sizeof(int));
        if (!nnz_per_row) {
            fprintf(stderr, "Errore di allocazione per nnz_per_row\n");
            exit(EXIT_FAILURE);
        }
        
        int block_maxnz = 0;
        /* Conto i non-zeri per ogni riga (senza modificare coo_index) */
        int temp_index = coo_index;
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
        
        /* Allocazione degli array JA e AS per il blocco corrente.
           Vengono allocati come array 1D di dimensione blockRows * block_maxnz. */
        int *JA = (int *) malloc(blockRows * block_maxnz * sizeof(int));
        double *AS = (double *) malloc(blockRows * block_maxnz * sizeof(double));
        if (!JA || !AS) {
            fprintf(stderr, "Errore di allocazione per gli array del blocco\n");
            exit(EXIT_FAILURE);
        }
        
        /* Inizializzo gli array: uso -1 in JA per indicare celle vuote e 0.0 in AS */
        for (int i = 0; i < blockRows * block_maxnz; i++) {
            JA[i] = -1;
            AS[i] = 0.0;
        }
        
        /* Riempio gli array JA e AS per ogni riga del blocco */
        for (int i = 0; i < blockRows; i++) {
            int currentRow = startRow + i;
            int count = 0;
            while (coo_index < coo->nz && coo->matrix[coo_index].row == currentRow) {
                int index = i * block_maxnz + count;
                JA[index] = coo->matrix[coo_index].col;
                AS[index] = coo->matrix[coo_index].value;
                count++;
                coo_index++;
            }
        }
        
        /* Popolo il blocco nella struttura HLL */
        hll->blocks[b].block_rows = blockRows;
        hll->blocks[b].N = coo->N;
        hll->blocks[b].maxnz = block_maxnz;
        hll->blocks[b].JA = JA;
        hll->blocks[b].AS = AS;
        
        free(nnz_per_row);
    }
    
    return hll;
}




