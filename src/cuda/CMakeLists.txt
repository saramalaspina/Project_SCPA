cmake_minimum_required(VERSION 3.8)

project(parallel_matrix_vector CUDA C)

# Abilita il supporto per CUDA in CMake
enable_language(CUDA)

# Cerca e trova il toolkit di CUDA installato sul sistema
find_package(CUDA REQUIRED)

# Imposta lo standard di C e CUDA
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# Imposta i flag di compilazione per C e CUDA
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# La directory per gli header
include_directories(../lib)
include_directories(../src/common)

# Riceve la lista dei file C da Makefile
set(COMMON_SOURCES ${COMMON_SRC})

# Definizione dei file CUDA
set(CUDA_SOURCES
    main.cu
    parallel_prod.cu
)

# Creazione dell'eseguibile
add_executable(exec_cuda ${COMMON_SOURCES} ${CUDA_SOURCES})

# Impostazioni specifiche per CUDA
set_target_properties(exec_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON   # Permette la compilazione separata dei file CUDA
    CUDA_ARCHITECTURES 75       # Target architettura CUDA, modifica se necessario
)

# Link alle librerie necessarie
target_link_libraries(exec_cuda PRIVATE cudart m)

# Specifica il linguaggio per i file sorgente
set_source_files_properties(${COMMON_SOURCES} PROPERTIES LANGUAGE C)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
