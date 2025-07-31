#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

python run_gns_dis.py \
    --exp_name "TGV/run0" \
    --seed 0 \
    --dataset_root "" \
    --data_save_path "" \
    --seq_length 7 \
    --split_interval 1 \
    --max_train_data 20000 \
    --connectivity_radius 0.029 \
    --dt 0.0004 \
    --learning_rate 1e-4 \
    --end_learning_rate 1e-5 \
    --num_epochs 200 \
    --batch_size 2 \
    --save_freq 10 \
    --to_train \
    --model_path "" \
    --output_size 2 \
    --latent_size 128 \
    --num_layers 4 \
    --message_passing_steps 10 \
    --particle_type_latent_size 32 \
    --node_input_size 10 \
    --edge_input_size 3
