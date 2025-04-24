#!/bin/bash

echo "Hello World"

module purge
module load cudatoolkit/12.8 anaconda3/2024.6
module list
conda --version

conda activate gemmpower
