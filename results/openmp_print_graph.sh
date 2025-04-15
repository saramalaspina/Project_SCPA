#!/bin/bash

# List of required modules
required_modules=("pandas" "matplotlib" "seaborn")  # Add your modules here

# Function to check if a module is installed
is_module_installed() {
    python3 -c "import $1" &>/dev/null  # Try to import the module
    return $?  # Return the status of the import command
}

# Function to install a module using pip3
install_module() {
    echo "Module $1 not found. Installing..."
    pip3 install $1  # Use pip3 to install the module
}

# Check and install missing modules
for module in "${required_modules[@]}"; do
    is_module_installed $module  # Check if the module is installed
    if [ $? -ne 0 ]; then  # If the module is not installed
        install_module $module  # Install the module
    else
        echo "Module $module is already installed."  # If the module is already installed
    fi
done

# Run Python scripts
scripts=("openmp_csr_hll_gflops.py" "openmp_csr_hll_speedup.py" "openmp_scheduling_totaltime.py" "openmp_scheduling.py" "openmp_threads_gflops.py" "openmp_threads_speedup.py")  # Add your Python scripts here

for script in "${scripts[@]}"; do
    echo "Running $script..."
    python3 $script  # Run the Python script with python3
done

