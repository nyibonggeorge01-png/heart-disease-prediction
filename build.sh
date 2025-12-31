#!/bin/bash
# Exit on error
set -e

echo "=== Setting up Python 3.9 environment ==="

# Ensure pip is up to date
echo "=== Upgrading pip ==="
python -m pip install --upgrade pip

# Install dependencies
echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Build completed successfully ==="
