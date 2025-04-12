#!/bin/bash

# Clear previous results
> results/cuda/performance.csv
> results/cuda/speedup.csv

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
    "cant.mtx"
    "olafu.mtx"
    "Cube_Coup_dt0.mtx"
    "ML_Laplace.mtx"
    "bcsstk17.mtx"
    "mac_econ_fwd500.mtx"
    "mhd4800a.mtx"
    "cop20k_A.mtx"
    "raefsky2.mtx"
    "af23560.mtx"
    "lung2.mtx"
    "PR02R.mtx"
    "FEM_3D_thermal1.mtx"
    "thermal1.mtx"
    "thermal2.mtx"
    "thermomech_TK.mtx"
    "nlpkkt80.mtx"
    "webbase-1M.mtx"
    "dc1.mtx"
    "amazon0302.mtx"
    "af_1_k101.mtx"
    "roadNet-PA.mtx"
)

echo "Loading modules for CUDA..."
module -s load gnu mpich cuda

echo "Compiling and running CUDA version..."
make cuda
if [ $? -ne 0 ]; then
    echo "Error during CUDA compilation"
    exit 1
fi

# Iterate over the matrix list
for MATRIX_PATH in "${MATRICES[@]}"; do
    echo "Running CUDA for $MATRIX_PATH..."
    ./bin/cuda "../matrix/$MATRIX_PATH"
done

echo "CUDA execution completed for all matrices."
