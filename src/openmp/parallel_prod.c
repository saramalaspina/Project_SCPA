#include "../../lib/utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


 // Performs sparse matrix-vector multiplication using the CSR format in parallel with OpenMP
void prod_openmp_csr(int M, const CSRMatrix * __restrict__ csr, const double * __restrict__ x, double * __restrict__ y, const int * __restrict__ row_bounds) {
    int num_threads = omp_get_max_threads();

    // Start a parallel region with a fixed number of threads
    #pragma omp parallel num_threads(num_threads)
    {
        // Get the thread ID
        int tid = omp_get_thread_num();

        // Each thread processes a specific range of rows, as defined in row_bounds
        for (int i = row_bounds[tid]; i < row_bounds[tid + 1]; i++) {
            double sum = 0.0;
            int start = csr->IRP[i];     
            int end   = csr->IRP[i + 1]; 

            for (int j = start; j < end; j++) {
                sum += csr->AS[j] * x[csr->JA[j]];
            }

            y[i] = sum;
        }
    }
}

// Performs sparse matrix-vector multiplication using the HLL format (with ELLPACK block) in parallel with OpenMP
void prod_openmp_hll(const HLLMatrix * __restrict__ hll, const double * __restrict__ x, double * __restrict__ y) {

    // Parallelize over blocks using OpenMP, with guided scheduling to balance workload dynamically
    #pragma omp parallel for schedule(guided)
    for (int b = 0; b < hll->numBlocks; b++) {
        EllpackBlock *block = &hll->blocks[b]; 
        int base = b * hll->hackSize;          // Base row index for this block

        // Iterate over rows in the block
        for (int i = 0; i < block->block_rows; i++) {
            int global_row = base + i;         // Compute the global row index
            double sum = 0.0;

            // Iterate over non-zero elements in the row (column-major access)
            for (int j = 0; j < block->maxnz; j++) {
                int idx = j * block->block_rows + i;  // Compute 1D index for column-major access
                int col = block->JA[idx];             // Column index

                // Check if the index is valid (not a padded entry)
                if (col != -1) {  
                    sum += block->AS[idx] * x[col]; 
                }
            }

            y[global_row] = sum;
        }
    }
}


// Performs sparse matrix-vector multiplication using the HLL format (with ELLPACK block) in parallel with OpenMP
void prod_openmp_hll_optimized(const HLLMatrix * __restrict__ hll, const double * __restrict__ x, double * __restrict__ y, const int * __restrict__ block_bounds) {
    int num_threads = omp_get_max_threads();

    // Start a parallel region with a fixed number of threads
    #pragma omp parallel num_threads(num_threads)
    {
        // Get the thread ID
        int tid = omp_get_thread_num();

        // Each thread works on a subset of blocks assigned to it
        for (int b = block_bounds[tid]; b < block_bounds[tid + 1]; b++) {
            EllpackBlock *block = &hll->blocks[b];
            int base = b * hll->hackSize;

            // Loop over the rows in the current block
            for (int i = 0; i < block->block_rows; i++) {
                int global_row = base + i;
                double sum = 0.0;

                // Precompute the base index for column-major access
                int row_offset = i;

                // Compute the dot product of the row with vector x
                for (int j = 0; j < block->maxnz; j++) {
                    int idx = j * block->block_rows + row_offset;
                    int col = block->JA[idx];
                    sum += block->AS[idx] * x[col];
                }

                // Store the result in the output vector
                y[global_row] = sum;
            }
        }
    }
 
}



    


