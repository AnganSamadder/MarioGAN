#!/bin/bash
# Script to compile the MarioRandomLevelViewer and its dependencies

# Clean previous build
echo "Removing old build directory..."
rm -rf marioaiDagstuhl/bin

# Ensure the output directory exists
mkdir -p marioaiDagstuhl/bin

# Find all Java source files
find marioaiDagstuhl/src -name "*.java" > sources.txt

# Compile all Java files found
echo "Compiling all source files..."
javac -sourcepath marioaiDagstuhl/src -cp 'marioaiDagstuhl/bin:marioaiDagstuhl/lib/*:marioaiDagstuhl/' -d marioaiDagstuhl/bin @sources.txt

# Check if compilation was successful
if [ $? -eq 0 ]; then
  echo "Compilation successful."
else
  echo "Compilation failed."
  # Clean up sources file even if failed
  rm sources.txt
  exit 1
fi

# Clean up the temporary file list
rm sources.txt

echo "Build complete." 