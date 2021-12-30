#!/bin/bash

# Install dependencies
echo -e "Installing dependencies if they haven't already been installed..."
python3 -m pip install --user numpy pandas sklearn pymongo tensorflow tensorflow-io
echo -e "Running main..."
python3 main.py