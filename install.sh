#!/bin/bash

echo -e "Installing dependencies if they haven't already been installed..."
python3 -m pip install --user numpy==1.16.4 pandas sklearn pymongo tensorflow tensorflow-io