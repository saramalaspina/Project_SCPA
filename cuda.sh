#!/bin/bash

> results/cuda/performance.csv
> results/cuda/speedup.csv

# Lista di path delle matrici
MATRICI=(
    "Cube_Coup_dt0.mtx"
)
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
    echo "Eseguendo CUDA per $MATRIX_PATH..."
    ./bin/cuda "../matrix/$MATRIX_PATH"
done

echo "Esecuzione CUDA completata per tutte le matrici."
