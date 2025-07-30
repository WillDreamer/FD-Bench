#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5


python main.py \
    --exp_name "NS1000/run2" \
    --seed 0 \
    --dataset_root "" \
    --data_save_path "" \
    --seq_length 1 \
    --learning_rate 1e-4 \
    --end_learning_rate 1e-6 \
    --num_epochs 100 \
    --batch_size 2 \
    --save_freq 10 \
    --to_train \
    --model_path "" \
    --modes 12 \
    --width 32 \
    --input_channels 1 \
    --output_channels 1
