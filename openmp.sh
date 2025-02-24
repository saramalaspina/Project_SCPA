#!/bin/bash

# Lista di path delle matrici
MATRICI=(
    "cage4.mtx"
    "mhda416.mtx"
    "mcfe.mtx"
    "olm1000.mtx"
    "adder_dcop_32.mtx"
    "west2021.mtx"
    "cavity10.mtx"
    "rdist2.mtx"
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
