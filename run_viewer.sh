#!/bin/bash

# Script to run MarioLevelViewer

# Ensure classes are compiled first (optional, uncomment if needed)
# ./compile_all.sh
# if [ $? -ne 0 ]; then exit 1; fi 

# Default Java options
JAVA_OPTS=""
# Process arguments
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-save)
      JAVA_OPTS="-Dmariogan.savefiles=false"
      shift # past argument
      ;;
    *)
      ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

echo "Running MarioLevelViewer..."

# Run the class, applying options and passing remaining arguments
# Note: Ensure ARGS[@] is last if latent vector is expected there
java $JAVA_OPTS -cp "marioaiDagstuhl/bin:marioaiDagstuhl/lib/*" viewer.MarioLevelViewer "${ARGS[@]}"

exit 0 