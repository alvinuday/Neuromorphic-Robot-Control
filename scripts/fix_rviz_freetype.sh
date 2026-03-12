#!/usr/bin/env bash
# Fix RViz2 "Library not loaded: libfreetype.6.dylib" on macOS when using conda/micromamba.
# Run this from a shell where ros_env (or your ROS conda env) is activated.

set -e

if [ -z "$CONDA_PREFIX" ]; then
  echo "Error: No conda env active. Run: micromamba activate ros_env"
  exit 1
fi

# Install freetype into the conda env so the library exists
echo "Installing freetype into $CONDA_PREFIX ..."
micromamba install -y -c conda-forge freetype

# Prepend conda env lib so rviz_ogre_vendor finds libfreetype.6.dylib
export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"

echo ""
echo "Freetype installed. To run RViz2 with the fix, use:"
echo "  export DYLD_LIBRARY_PATH=\"${CONDA_PREFIX}/lib:\${DYLD_LIBRARY_PATH:-}\""
echo "  rviz2"
echo ""
echo "Or run this script and then rviz2 in the same shell:"
echo "  source scripts/fix_rviz_freetype.sh"
echo "  rviz2"
echo ""

# If rviz2 was passed as argument, run it
if [ "$1" = "rviz2" ]; then
  exec rviz2 "${@:2}"
fi
