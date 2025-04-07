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

CSRMatrix *convert_coo_to_csr(COOElement *coo, int nz, int m);
HLLMatrix *convert_coo_to_hll(MatrixElement *coo, int hackSize);

//utils 
double *generate_vector(const char *matrix_name, int N);
void free_hll_matrix(HLLMatrix *hll);
void free_csr_matrix(CSRMatrix *csr);
void print_result(double *y, int M);
void calculate_performance_openmp(double *times, MatrixElement *mat, char *matrix_name, char *type, int numThreads, double *time, char* filename);
void calculate_performance_cuda(double *times, MatrixElement *mat, const char *matrix_name, const char *type, double *time);
int check_results(double *y_serial, double *y_parallel, int size);
void calculate_speedup(const char* matrix_name, double time_serial, double time_csr, double time_hll, const char* file, int numThreads, int nz);
int compare_coo(const void *a, const void *b);
void compute_row_bounds(CSRMatrix *csr, int M, int num_threads, int *row_bounds);

// reader

MatrixElement* read_matrix(char* filename);

// serial product

void prod_serial(int M, CSRMatrix *csr, double *x, double *y);

// parallel product openmp

void prod_openmp_csr(int M, CSRMatrix *csr, double *x, double *y, int *row_bounds);
void prod_openmp_hll(HLLMatrix *hll, double *x, double *y);

// parallel product cuda

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

void prod_cuda_csr(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time);
void prod_cuda_hll(const HLLMatrix *hllHost, const double *xHost, double *yHost, int totalRows, float *elapsed_time);

#define REPETITIONS 30

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
