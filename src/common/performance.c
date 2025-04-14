#include "../../lib/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

int file_is_empty(FILE *fp) {
    fseek(fp, 0, SEEK_END);  
    long size = ftell(fp);  
    return size == 0;
}

void calculate_performance_openmp(double *times, MatrixElement *mat, char *matrix_name, char *type, int numThreads, double *time, char* filename){
    double total_time = 0.0;

    // Sum execution time in each repetition, excluding the first ten due to initial overhead factors
    for (int i = 10; i < REPETITIONS; i++){
        total_time += times[i];
    }

    // Calculate average execution time (in seconds) 
    total_time /= (REPETITIONS - 10); 
    double time_ms = total_time * 1000;

    // Calculate average flops and gigaflops
    double flops = (2.0 * mat->nz) / total_time;
    double gflops = flops / 1e9;

    printf("Risultati: gflops %.17g\n", gflops);

    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(1);
    }

    if (file_is_empty(fp)) {
        fprintf(fp, "matrix, M, N, nz, type, avgTime, avgGFlops, nThreads\n");
    }
    fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f, %d\n", matrix_name, mat->M, mat->N, mat->nz, type, time_ms, gflops, numThreads);

    fclose(fp);    

    *time = time_ms;

}


void calculate_performance_cuda(double *times, MatrixElement *mat, const char *matrix_name, const char *type, double *time, char *filename){
    double total_time = 0.0;

    // Sum execution time in each repetition, excluding the first ten due to initial overhead factors
 
    for (int i = 10; i < REPETITIONS; i++){
        total_time += times[i];
    }

    // Calculate the average execution time (in milliseconds) 
    total_time /= (REPETITIONS - 10); 

    // Calculate average flops and gigaflops
    double flops = (2.0 * mat->nz) / (total_time/1000);
    double gflops = flops / 1e9;

    printf("Risultati: gflops %.17g\n", gflops);

    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        exit(1);
    }
  
    if (file_is_empty(fp)) {
        fprintf(fp, "matrix, M, N, nz, type, avgTime, avgGFlops\n");
    }
    fprintf(fp, "%s, %d, %d, %d, %s, %.6f, %.6f\n", matrix_name, mat->M, mat->N, mat->nz, type, total_time, gflops);

    fclose(fp);    

    *time = total_time;

}

void calculate_speedup(const char* matrix_name, double time_serial, double time_csr, double time_hll, const char* file, int numThreads, int nz){

    // Calculate the speedup of parallel execution compared to serial execution
    double speedup_csr = time_serial/time_csr;
    double speedup_hll = time_serial/time_hll;

    printf("Speedup csr %.17g\n", speedup_csr);
    printf("Speedup hll %.17g\n", speedup_hll);

    FILE *fp = fopen(file, "a");
    if (fp == NULL) {
        perror("Errore nell'apertura del file CSV");
        return;
    }

    if(numThreads != 0){
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, nz, time_serial, speedup_csr, speedup_hll, nThreads\n");
        }
        fprintf(fp, "%s, %d, %.6f, %.6f, %.6f, %d\n", matrix_name, nz, time_serial, speedup_csr, speedup_hll, numThreads);
    } else {
        if (file_is_empty(fp)) {
            fprintf(fp, "matrix, nz, time_serial, speedup_csr, speedup_hll\n");
        }
        fprintf(fp, "%s, %d, %.6f, %.6f, %.6f\n", matrix_name, nz, time_serial, speedup_csr, speedup_hll);
    }
   
    fclose(fp);    
    
}