#!/bin/bash

# Script to run MarioLevelViewer

# Ensure classes are compiled first (optional, uncomment if needed)
# ./compile_all.sh
# if [ $? -ne 0 ]; then exit 1; fi 

# Default Java options
JAVA_OPTS="-Djava.awt.headless=true" # Always start with headless mode enabled
CHECKPOINT_PATH_VAL=""
GENERATOR_SCRIPT_VAL="" # Variable to hold generator script path
export MARIOGAN_CHECKPOINT="" # Exported environment variable
LATENT_FILE_VAL="" # Variable to hold latent filename path

# Process arguments
POS_ARGS=() # Store positional arguments separately
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-save)
      JAVA_OPTS="$JAVA_OPTS -Dmariogan.savefiles=false" # Append the no-save flag if provided
      shift # past argument
      ;;
    --checkpoint)
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        # Store path for Java property AND export for Python
        CHECKPOINT_PATH_VAL="$2"
        export MARIOGAN_CHECKPOINT="$CHECKPOINT_PATH_VAL"
        shift 2 # past argument and value
      else
        echo "Error: --checkpoint requires a path argument." >&2
        exit 1
      fi
      ;;
    --generator-script) # New argument
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        GENERATOR_SCRIPT_VAL="$2"
        shift 2 # past argument and value
      else
        echo "Error: --generator-script requires a path argument." >&2
        exit 1
      fi
      ;;
    -lf) # Handle latent file argument
      if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
        LATENT_FILE_VAL="$2"
        shift 2 # past argument and value
      else
        echo "Error: -lf requires a filename argument." >&2
        exit 1
      fi
      ;;
    *)
      POS_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# Append checkpoint argument as Java system property if it exists
if [[ -n "$CHECKPOINT_PATH_VAL" ]]; then
    # Pass via system property as well, in case Java side uses it
    JAVA_OPTS="$JAVA_OPTS -Dmariogan.checkpoint=$CHECKPOINT_PATH_VAL"
fi

# Append generator script argument as Java system property if it exists
if [[ -n "$GENERATOR_SCRIPT_VAL" ]]; then
    JAVA_OPTS="$JAVA_OPTS -Dmariogan.generatorScript=$GENERATOR_SCRIPT_VAL"
fi

# Prepend -lf argument if present before other positional args
JAVA_ARGS=()
if [[ -n "$LATENT_FILE_VAL" ]]; then
    JAVA_ARGS+=("-lf" "$LATENT_FILE_VAL")
fi

# Append remaining positional arguments
JAVA_ARGS+=("${POS_ARGS[@]}")

echo "Running MarioLevelViewer with Java Options: $JAVA_OPTS"
echo "Exporting MARIOGAN_CHECKPOINT=${MARIOGAN_CHECKPOINT}"

# Run the class, applying options and passing processed arguments
# Note: Ensure POS_ARGS[@] is last if latent vector is expected there
java $JAVA_OPTS -cp "marioaiDagstuhl/bin:marioaiDagstuhl/lib/*" viewer.MarioLevelViewer "${JAVA_ARGS[@]}"

exit 0 