#!/bin/bash

echo "========================================"
echo "Pushing to skyeAssetManagement/pythonBT"
echo "========================================"
echo

# First, ensure the remote is set correctly
echo "Setting up remote..."
git remote remove skyeasset 2>/dev/null
git remote add skyeasset https://github.com/skyeAssetManagement/pythonBT.git

echo
echo "Pushing main branch (original OMtree code)..."
git push skyeasset main

echo
echo "Pushing merge-with-ABToPython branch (integrated system)..."
git push skyeasset merge-with-ABToPython

echo
echo "========================================"
echo "Push complete!"
echo "========================================"
echo
echo "Both branches have been pushed:"
echo "- main: Original OMtree code"
echo "- merge-with-ABToPython: Integrated system with ABtoPython"
echo
echo "You can now go to GitHub and:"
echo "1. Set the default branch if needed"
echo "2. Create a Pull Request to merge branches"
echo "3. Configure branch protection rules"
echo