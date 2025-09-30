# FD-Bench/train_public.sh
#!/bin/bash

IF_BASELINE=True
MODULE_NAME="fdbench.models.baselines"
MODEL_NAME="fno"

CONFIG_PATH="config/public/${MODEL_NAME}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: The config ${MODEL_NAME}.yaml does not exist!"
    exit 1
fi

export WANDB_ENTITY="FD-Bench"
export WANDB_PROJECT="${WANDB_ENTITY}_Baselines"
export WANDB_NAME="${MODULE_NAME}_${MODEL_NAME}_${REMARK}"
# export WANDB_API_KEY="xxxxxxx"

accelerate launch --main_process_port 29514 src/train.py \
    --config_file "$CONFIG_PATH" \
    --remark "$REMARK" \
    --if_baseline "$IF_BASELINE" \
    --pub_module_name "$MODULE_NAME" \
    --pub_model_name "$MODEL_NAME"