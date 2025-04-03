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
	@echo "Compilazione versione OpenMP..."
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o bin/openmp $(COMMON_SRC) $(OPENMP_SRC)

# CUDA Compilation using CMake
cuda: 
	@echo "Pulizia della cartella build..."
	rm -rf $(BUILD_DIR)
	@echo "Compilazione versione CUDA con CMake..."
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ../src/cuda && make || echo "Errore nella compilazione CUDA!"
	cp $(BUILD_DIR)/exec_cuda bin/cuda || echo "Errore: eseguibile CUDA non trovato!"


clean:
	@echo "Pulizia dei file compilati..."
	rm -rf bin/openmp bin/cuda $(BUILD_DIR)


.PHONY: all openmp cuda clean

