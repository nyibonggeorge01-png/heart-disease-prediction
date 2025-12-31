#!/bin/bash
# vercel-build.sh - Custom build script for Vercel deployment

set -e  # Exit immediately if a command exits with a non-zero status

# Print environment information
echo "=== Vercel Build Script ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version 2>&1 || echo 'Python not found')"
echo "Pip version: $(pip --version 2>&1 || echo 'Pip not found')"
echo "Python path: $(which python 2>/dev/null || echo 'Not found')"

# Install dependencies
echo -e "\n=== Installing Dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# Verify installations
echo -e "\n=== Verifying Installations ==="
pip list

echo -e "\n=== Build Completed Successfully ==="