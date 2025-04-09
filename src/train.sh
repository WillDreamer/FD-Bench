# FD-Bench/train.sh
#!/bin/bash

# SPATIAL_REP="self_atten"
# SPATIAL_REP="conv"
# SPATIAL_REP="graph"
# SPATIAL_REP='latent'
SPATIAL_REP='fourier'

TEMPORAL_REP="next_step"

TARGET="variable"
CONFIG_PATH="config/${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml"

# REMARK="ViTL_Dim_768_Epoch_2k_Cyc_up_1k_down_1k_lr_1e-3"
# REMARK='cyc_1k_up_1k_down_1k_N_30_hidden_256_epo2k_grad_layer_8'
REMARK='KF_data'

export WANDB_ENTITY="FD-Bench"
export WANDB_PROJECT="${WANDB_ENTITY}_${TARGET}"
export WANDB_NAME="${SPATIAL_REP}_${TEMPORAL_REP}_${REMARK}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"

# 运行训练
cd /wanghaixin/FD-Bench
/root/anaconda3/bin/accelerate launch --main_process_port 29513 src/train.py \
    --config "$CONFIG_PATH" \
    --remark "$REMARK" \