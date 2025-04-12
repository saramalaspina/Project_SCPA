# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.8)

# Define the project name and the supported languages
project(parallel_matrix_vector CUDA C)

# Enable CUDA language support
enable_language(CUDA)

# Set the C and CUDA language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

# Add the directory containing header files to the include path
include_directories(../lib)

# Search for all .c source files in the 'common' directory
file(GLOB COMMON_SOURCES ../common/*.c)

# Define the CUDA source files for the project
set(CUDA_SOURCES
    main.cu
    parallel_prod.cu
)

# Create the executable named 'exec_cuda' from the specified sources
add_executable(exec_cuda ${COMMON_SOURCES} ${CUDA_SOURCES})

# CUDA-specific settings
set_target_properties(exec_cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Link the executable with necessary libraries
target_link_libraries(exec_cuda PRIVATE cudart m)

# Explicitly set the language for the source files
set_source_files_properties(${COMMON_SOURCES} PROPERTIES LANGUAGE C)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Set compile options for both CUDA and C files
target_compile_options(exec_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler=-Wall -Xcompiler=-Wextra>
    $<$<COMPILE_LANGUAGE:C>:-O3 -Wall -Wextra>
)
