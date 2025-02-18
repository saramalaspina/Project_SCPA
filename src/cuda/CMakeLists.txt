cmake_minimum_required(VERSION 3.8)

project(parallel_matrix_vector CUDA C)

# Abilita il supporto per CUDA
enable_language(CUDA)

# Imposta lo standard di C e CUDA
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# La directory per gli header
include_directories(../lib)

# Cerca tutti i file .c nella cartella common
file(GLOB COMMON_SOURCES ../common/*.c)

# Definizione dei file CUDA
set(CUDA_SOURCES
    main.cu
    parallel_prod.cu
    execution.cu
)

# Creazione dell'eseguibile
add_executable(exec_cuda ${COMMON_SOURCES} ${CUDA_SOURCES})

# Impostazioni specifiche per CUDA
set_target_properties(exec_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75  # Modifica se necessario
)

# Link alle librerie necessarie
target_link_libraries(exec_cuda PRIVATE cudart m)

# Specifica il linguaggio per i file sorgente
set_source_files_properties(${COMMON_SOURCES} PROPERTIES LANGUAGE C)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Ottimizzazione e warning per file CUDA e C
target_compile_options(exec_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall -Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Wextra>
)

