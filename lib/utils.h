#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#define REPETITIONS 30

//converter
#define HACKSIZE 32

//COO Matrix
typedef struct {
    int row, col;
    double value;
} COOElement;

typedef struct {
    int M, N, nz;
    COOElement *matrix;
} MatrixElement;

//HLL Matrix with Ellpack Block
typedef struct {
    int block_rows;   
    int N;           
    int maxnz;   
    int *JA;       
    double *AS;  
} EllpackBlock;

typedef struct {
    int hackSize;     
    int numBlocks;  
    EllpackBlock *blocks;
} HLLMatrix;

//CSR Matrix
typedef struct {
    int *IRP;
    int *JA;
    double *AS;
} CSRMatrix;

CSRMatrix *convert_coo_to_csr(COOElement *coo, int nz, int m);
HLLMatrix *convert_coo_to_hll(MatrixElement *coo, int hackSize);

//performance
void calculate_performance_openmp(double *times, MatrixElement *mat, char *matrix_name, char *type, int numThreads, double *time, char* filename);
void calculate_performance_cuda(double *times, MatrixElement *mat, const char *matrix_name, const char *type, double *time,  char *filename);
void calculate_speedup(const char* matrix_name, double time_serial, double time_csr, double time_hll, const char* file, int numThreads, int nz);

//utils 
double *generate_vector(int N);
void free_hll_matrix(HLLMatrix *hll);
void free_csr_matrix(CSRMatrix *csr);
void print_result(double *y, int M);
int check_results(double *y_serial, double *y_parallel, int size);
int compare_coo(const void *a, const void *b);
void compute_row_bounds(CSRMatrix *csr, int M, int num_threads, int *row_bounds);
void compute_block_bounds(int numBlocks, int num_threads, int *block_bounds);


//reader
MatrixElement* read_matrix(char* filename);

//serial product
void prod_serial(int M, CSRMatrix *csr, double *x, double *y);

//parallel product openmp
void prod_openmp_csr(int M, const CSRMatrix * __restrict__ csr, const double * __restrict__ x, double * __restrict__ y, const int * __restrict__ row_bounds);
void prod_openmp_hll(const HLLMatrix * __restrict__ hll, const double * __restrict__ x, double * __restrict__ y);
void prod_openmp_hll_optimized(const HLLMatrix * __restrict__ hll, const double * __restrict__ x, double * __restrict__ y, const int * __restrict__ block_bounds);

//parallel product cuda
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

void prod_cuda_csr(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time);
void prod_cuda_csr_warp(int M, int N, CSRMatrix *csr, double *x, double *y, float *elapsed_time);
void prod_cuda_hll(const HLLMatrix *hllHost, const double *xHost, double *yHost, int totalRows, float *elapsed_time);
void prod_cuda_hll_warp(const HLLMatrix *hllHost, const double *xHost, double *yHost, int totalRows, float *elapsed_time);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
