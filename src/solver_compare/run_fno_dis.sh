#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

python run_fno_dis.py \
    --exp_name "TGV/run0" \
    --seed 0 \
    --dataset_root "" \
    --data_save_path "" \
    --seq_length 7 \
    --split_interval 1 \
    --max_train_data 20000 \
    --stats_path "" \
    --learning_rate 1e-4 \
    --end_learning_rate 1e-5 \
    --num_epochs 200 \
    --batch_size 2 \
    --save_freq 10 \
    --to_train \
    --model_path "" \
    --modes 32 \
    --width 128 \
    --input_channels 10 \
    --output_channels 2 
