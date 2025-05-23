CC = gcc
CFLAGS = -O2 -Wall -Ilib
OPENMP_FLAGS = -fopenmp -lm 

COMMON_SRC = $(wildcard src/common/*.c) 
OPENMP_SRC = $(wildcard src/openmp/*.c)
OPENMP_MAIN = src/openmp/main.c

BUILD_DIR = build

all: openmp cuda

bin:
	mkdir -p bin

openmp: bin $(COMMON_SRC) $(OPENMP_SRC) $(OPENMP_MAIN)
	@echo "Compiling OpenMP version..."
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o bin/openmp $(COMMON_SRC) $(OPENMP_SRC)

# CUDA Compilation using CMake
cuda: 
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "Compiling CUDA version with CMake..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ../src/cuda && make || echo "Error during CUDA compilation!"
	cp $(BUILD_DIR)/exec_cuda bin/cuda || echo "Error: CUDA executable not found!"

clean:
	@echo "Cleaning compiled files..."
	rm -rf bin/openmp bin/cuda $(BUILD_DIR)

.PHONY: all openmp cuda clean


