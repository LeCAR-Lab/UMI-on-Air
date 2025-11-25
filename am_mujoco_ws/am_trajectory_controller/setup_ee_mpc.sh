#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set ACADOS paths relative to script location
export ACADOS_SOURCE_DIR="$SCRIPT_DIR/acados"
export ACADOS_PYTHON_INTERFACE_PATH="$SCRIPT_DIR/acados/interfaces/acados_template/acados_template"
export LD_LIBRARY_PATH="$SCRIPT_DIR/acados/lib:$LD_LIBRARY_PATH"

echo "Fixed env: ACADOS_SOURCE_DIR=$ACADOS_SOURCE_DIR"
echo "ACADOS_PYTHON_INTERFACE_PATH=$ACADOS_PYTHON_INTERFACE_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
