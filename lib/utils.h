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

/* Struttura per un blocco in formato ELLPACK */
typedef struct {
    int block_rows;   // Numero di righe del blocco (<= hackSize)
    int N;            // Numero di colonne della matrice (uguale per tutte)
    int maxnz;        // Massimo numero di non-zeri in una riga di questo blocco
    int *JA;          // Array 1D (di dimensione block_rows*maxnz) degli indici di colonna
    double *AS;       // Array 1D (di dimensione block_rows*maxnz) dei valori non-zero
} EllpackBlock;

/* Struttura per la matrice in formato HLL */
typedef struct {
    int hackSize;        // Parametro HackSize (es. 32)
    int numBlocks;       // Numero di blocchi (M_totale/hackSize, ultimo blocco eventualmente ridotto)
    EllpackBlock *blocks;// Array di blocchi in formato ELLPACK
} HLLMatrix;

typedef struct {
    int *IRP;
    int *JA;
    double *AS;
} CSRMatrix;

CSRMatrix *convertCOOtoCSR(COOElement *coo, int nz, int m);
HLLMatrix *convertCOOtoHLL(MatrixElement *coo, int hackSize);

//utils 
double *generateVector(int N);
void freeHLLMatrix(HLLMatrix *hll);
void freeCSRMatrix(CSRMatrix *csr);
void printResult(double *y, int M);
void calculatePerformanceOpenMP(double *times, MatrixElement *mat, char *matrix_name, char *type, int numThreads, double *time);
void calculatePerformanceCuda(double *times, MatrixElement *mat, char *matrix_name, char *type, double *time);
int checkResults(double* arr1, double* arr2, int n);
void calculateSpeedup(char *matrix_name, double time_serial, double time_csr, double time_hll, char *paral, int numThreads);
int compareCOO(const void *a, const void *b);

// reader

MatrixElement* read_matrix(char* filename);

// serial product

void prodSerial(int M, CSRMatrix *csr, double *x, double *y);

// parallel product openmp

void prodOpenmpCSR(int M, CSRMatrix *csr, double *x, double *y);
void prodOpenmpHLL(HLLMatrix *hll, double *x, double *y);

// parallel product cuda

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

void prodCudaCSR(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time);
void prodCudaHLL(const HLLMatrix *hllHost, const double *xHost, double *yHost, int totalRows, float *elapsed_time);

#define REPETITIONS 5

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
