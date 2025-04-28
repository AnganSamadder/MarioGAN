#!/bin/bash

# Script to compile all Java files in the project

echo "Compiling Java source files..."

# Assuming necessary JARs are in marioaiDagstuhl/lib/
javac -d marioaiDagstuhl/bin -sourcepath marioaiDagstuhl/src -cp "marioaiDagstuhl/lib/*" $(find marioaiDagstuhl/src -name "*.java")

# Check if compilation was successful
if [ $? -eq 0 ]; then
  echo "Compilation successful."
else
  echo "Compilation failed."
  exit 1
fi

exit 0 