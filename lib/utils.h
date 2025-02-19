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
void trasponseHLLMatrix(HLLMatrix *hll);
void printHLLMatrixTransposed(const HLLMatrix *H);

//utils 
double *generateVector(int N);
void freeHLLMatrix(HLLMatrix *hll);
void freeCSRMatrix(CSRMatrix *csr);
void printResult(double *y, int M);
void calculatePerformance(double *times, MatrixElement *mat, char *matrix_name, char *type, char *paral, int numThreads);
int checkResults(double* arr1, double* arr2, int n);

// reader

MatrixElement* read_matrix(char* filename);

// mediator

typedef struct {
    void (*convertToCSR)(COOElement*, int, int, CSRMatrix*);
    void (*convertToHLL)(COOElement*, int, int, int, HLLMatrix*);
} MatrixConversionMediator;

MatrixConversionMediator createMatrixMediator();

// serial product

void prodSerial(int M, CSRMatrix *csr, double *x, double *y);

// parallel product openmp

void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y);
void prodOpenmpHLL(HLLMatrix *hll, double *x, double *y);

// parallel product cuda

#define THREADS_PER_BLOCK 256

void prodCudaCSR(int M, int N, int *IRP, int *JA, double *AS, double *x, double *y, float *elapsed_time);
void prodCudaHLL(const HLLMatrix *hll, int total_rows, int total_cols, const double *x,  double *y, float *elapsed_time);

#define REPETITIONS 5

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
