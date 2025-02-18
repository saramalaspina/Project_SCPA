#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

// converter

#define HACKSIZE 32

// Definizione COO
typedef struct {
    int row, col;
    double value;
} COOElement;

typedef struct {
    int M, N, nz;
    COOElement *matrix;
} MatrixElement;

typedef struct {
    int rows;
    int cols;
    int max_nz;
    int **JA;
    double **AS;
    int *JA_t;
    double *AS_t;
} ELLBlock;

typedef struct {
    int num_blocks;
    ELLBlock *blocks;
} HLLMatrix;

typedef struct {
    int *IRP;
    int *JA;
    double *AS;
} CSRMatrix;

void convertCOOtoCSR(COOElement *coo, int nz, int m, CSRMatrix *matrix);
void convertCOOtoHLL(COOElement *coo, int nz, int M, int N, HLLMatrix *hll);
void freeHLLMatrix(HLLMatrix *hll);
void freeCSRMatrix(CSRMatrix *csr);
void trasponseHLLMatrix(HLLMatrix *hll);
//void printHLLMatrixTransposed(const HLLMatrix *H);

// reader

MatrixElement* read_matrix(char* filename);

// mediator

typedef struct {
    void (*convertToCSR)(COOElement*, int, int, CSRMatrix*);
    void (*convertToHLL)(COOElement*, int, int, int, HLLMatrix*);
} MatrixConversionMediator;

MatrixConversionMediator createMatrixMediator();

// serial product

double *spmv_csr(int M, CSRMatrix *csr, double *x);

// parallel product openmp

double *spmv_csr_parallel(int M, CSRMatrix *csr, double *x);
double *spmv_hll_parallel(HLLMatrix *hll, double *x);

// parallel product cuda

#define THREADS_PER_BLOCK 256

double *spmv_csr_cuda(int M, int N, int *IRP, int *JA, double *AS, double *x);
double *spmv_hll_cuda(const HLLMatrix *hll, int total_rows, int total_cols, const double *x);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
