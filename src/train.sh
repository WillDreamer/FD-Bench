# FD-Bench/train.sh
#!/bin/bash

SPATIAL_REP='self_atten'
# SPATIAL_REP="graph"
# SPATIAL_REP='latent'
# SPATIAL_REP='fourier'
# SPATIAL_REP='conv'
# SPATIAL_REP='diffusion'
# SPATIAL_REP='flow'

# TEMPORAL_REP='self_atten'
TEMPORAL_REP="next_step"
# TEMPORAL_REP="temporal_bundling"
# TEMPORAL_REP="auto_regressive"
# TEMPORAL_REP="node"

TARGET="variable"
# TARGET="noise"
# TARGET='flow'

#! Important for log name
# REMARK='SA_Residual_CosineAnnealingWarmRestarts'
REMARK='lambdalr_DR'
# REMARK='lambdalr_train_for_rollout'

CONFIG_PATH="/wanghaixin/FD-Bench/config/${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: The modular combination ${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml does not exist!"
    exit 1
fi

export WANDB_ENTITY="FD-Bench"
export WANDB_PROJECT="${WANDB_ENTITY}_${TARGET}"
export WANDB_NAME="${SPATIAL_REP}_${TEMPORAL_REP}_${REMARK}"
export WANDB_API_KEY="ba70fcbc92808cc7a1750dd80ac3908295e6854f"

# 运行训练
cd /wanghaixin/FD-Bench
/root/anaconda3/bin/accelerate launch --main_process_port 29514 src/train.py \
    --config_file "$CONFIG_PATH" \
    --remark "$REMARK" \