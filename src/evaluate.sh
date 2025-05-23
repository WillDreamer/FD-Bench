SPATIAL_REP='fourier'
TEMPORAL_REP="next_step"
TARGET="variable"
EXP_NAME="PDE_CNS_fourier_next_step_variable_lambdalr_Rollout_20_0519-18:09"
RESUME_STEP=138000


CONFIG_PATH="/wanghaixin/FD-Bench/config/test/${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml"

cd /wanghaixin/FD-Bench
/root/anaconda3/bin/accelerate launch --main_process_port 29513 src/evaluate.py \
    --config "$CONFIG_PATH" \
    --remark "$REMARK" \
    --exp_name "$EXP_NAME" \
    --resume_step "$RESUME_STEP" \