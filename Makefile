CC = gcc
CFLAGS = -O2 -Wall -Ilib
OPENMP_FLAGS = -fopenmp 

COMMON_SRC = $(wildcard src/common/*.c) 
OPENMP_SRC = $(wildcard src/openmp/*.c) 
OPENMP_MAIN = src/openmp/main.c

BUILD_DIR = build

all: openmp cuda

bin:
	mkdir -p bin

# OpenMP Compilation
openmp: bin $(COMMON_SRC) $(OPENMP_MAIN) $(OPENMP_SRC)
	@echo "Compilazione versione OpenMP..."
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o bin/openmp $(COMMON_SRC) $(OPENMP_SRC)

# CUDA Compilation using CMake
cuda: clean
	@echo "Compilazione versione CUDA con CMake..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. && make

clean:
	@echo "Pulizia dei file compilati..."
	rm -rf bin/openmp bin/cuda $(BUILD_DIR)

run_cuda: cuda
	./$(BUILD_DIR)/exec_cuda $(MAT) $(MODE)

.PHONY: all openmp cuda clean
