#!/bin/bash
set -e

echo "=== Vercel Build Script ==="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

echo "=== Installed Packages ==="
pip list

echo "=== Build Completed Successfully ==="