#!/bin/bash

# Script to run CMAMarioSolver

# Ensure classes are compiled first (optional, uncomment if needed)
# ./compile_all.sh
# if [ $? -ne 0 ]; then exit 1; fi

echo "Running CMAMarioSolver..."

# Run the class, passing any command-line arguments ($@)
java -cp "marioaiDagstuhl/bin:marioaiDagstuhl/lib/*" cmatest.CMAMarioSolver "$@"

exit 0 