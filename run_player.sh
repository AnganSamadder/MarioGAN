#!/bin/bash

# Script to run MarioLevelPlayer

# Ensure classes are compiled first (optional, uncomment if needed)
# ./compile_all.sh
# if [ $? -ne 0 ]; then exit 1; fi

echo "Running MarioLevelPlayer..."

# Run the class, passing any command-line arguments ($@)
java -cp "marioaiDagstuhl/bin:marioaiDagstuhl/lib/*" viewer.MarioLevelPlayer "$@"

exit 0 