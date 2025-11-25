#!/bin/bash

name=$1   # first command-line argument
timestamp=$(date +"%d_%H_%M_%S")
work_dir="./work_dir/${name}_${timestamp}"

python main.py --work-dir "$work_dir" --mode train
python main.py --work-dir "$work_dir" --mode test