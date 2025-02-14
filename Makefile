CC = gcc 
NVCC = nvcc
CFLAGS = -O2 -Wall -Ilib
OPENMP_FLAGS = -fopenmp 
CUDA_FLAGS = -O2

COMMON_SRC = $(wildcard src/common/*.c) 
OPENMP_SRC = $(wildcard src/openmp/*.c) 
CUDA_SRC = $(wildcard src/cuda/*.cu)

OPENMP_MAIN = src/openmp/main.c

all: openmp cuda

bin:
	mkdir -p bin

# Compilazione della versione OpenMP
openmp: bin $(COMMON_SRC) $(OPENMP_MAIN) $(OPENMP_SRC)
	@echo "Compilazione versione OpenMP..."
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o bin/openmp $(COMMON_SRC) $(OPENMP_SRC)

# Compilazione della versione CUDA (solo se richiesto esplicitamente)
cuda: bin $(COMMON_SRC) $(CUDA_SRC)
	@echo "Compilazione versione CUDA..."
	$(NVCC) $(CUDA_FLAGS) -Iinclude -o bin/cuda $(COMMON_SRC) $(CUDA_SRC)

# Pulizia degli eseguibili
clean:
	@echo "Pulizia dei file compilati..."
	rm -f bin/openmp bin/cuda

.PHONY: all openmp cuda clean

