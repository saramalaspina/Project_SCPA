#!/bin/bash

> results/openmp/performance.csv
> results/openmp/speedup.csv

# Lista di path delle matrici
MATRICI=(
    "Cube_Coup_dt0.mtx"
)

echo "Compilazione ed esecuzione OpenMP..."
make openmp
if [ $? -ne 0 ]; then
    echo "Errore nella compilazione OpenMP"
    exit 1
fi

# Itera sulla lista delle matrici
for MATRIX_PATH in "${MATRICI[@]}"; do
    echo "Eseguendo OpenMP per $MATRIX_PATH..."
    ./bin/openmp "../matrix/$MATRIX_PATH"
done

echo "Esecuzione OpenMP completata per tutte le matrici."
