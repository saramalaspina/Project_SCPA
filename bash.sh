#!/bin/bash

> results/openmp/performance.csv
> results/openmp/speedup.csv

> results/cuda/performance.csv
> results/cuda/speedup.csv

# Lista di path delle matrici
MATRICI=(
    "cage4.mtx"
)

echo "Compilazione ed esecuzione OpenMP..."
make openmp
if [ $? -ne 0 ]; then
    echo "Errore nella compilazione OpenMP"
    exit 1
fi

echo "Caricamento moduli per CUDA..."
module -s load gnu mpich cuda

echo "Compilazione ed esecuzione CUDA..."
make cuda
if [ $? -ne 0 ]; then
    echo "Errore nella compilazione CUDA"
    exit 1
fi

# Itera sulla lista delle matrici
for MATRIX_PATH in "${MATRICI[@]}"; do
    echo "Eseguendo OpenMP per $MATRIX_PATH..."
    ./bin/openmp "../matrix/$MATRIX_PATH"

    echo "Eseguendo CUDA per $MATRIX_PATH..."
    ./bin/cuda "../matrix/$MATRIX_PATH"
done

echo "Esecuzione completata per tutte le matrici."
