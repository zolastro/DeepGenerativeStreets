#!/bin/bash
#$ -cwd
#$ -V
#$ -l coproc_p100=1
#$ -l h_rt=24:00:00
#$ -m be
# Prepare environment
module load cuda
# Python script with all parameters
conda activate deep-streets
python model.py
