#!/bin/bash
# Script to run MarioLevelViewer

# Run the compiled Java class in headless mode
java -Djava.awt.headless=true -cp 'marioaiDagstuhl/bin:marioaiDagstuhl/lib/*:marioaiDagstuhl/' viewer.MarioLevelViewer
 
echo "Execution finished." 