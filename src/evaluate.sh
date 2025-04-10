SPATIAL_REP='self_atten'
TEMPORAL_REP="next_step"
TARGET="variable"
EXP_NAME="PDE_CNS_self_atten_next_step_variable_ViTL_Dim_768_Epoch_2k_Cyc_up_2k_lr_1e-3_0403-18:51"


CONFIG_PATH="config/test/${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml"

cd /wanghaixin/FD-Bench
/root/anaconda3/bin/accelerate launch --main_process_port 29513 src/evaluate.py \
    --config "$CONFIG_PATH" \
    --remark "$REMARK" \
    --exp_name "$EXP_NAME"