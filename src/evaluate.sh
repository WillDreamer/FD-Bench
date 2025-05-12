SPATIAL_REP='self_atten'
TEMPORAL_REP="next_step"
TARGET="variable"
EXP_NAME="/wanghaixin/FD-Bench/exps/PDE_CNS_self_atten_next_step_variable_lambdalr_0509-20:22"
RESUME_STEP=28000

IF_ROLLOUT=True
ROLL_STEP=5

CONFIG_PATH="config/test/${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml"

cd /wanghaixin/FD-Bench
/root/anaconda3/bin/accelerate launch --main_process_port 29513 src/evaluate.py \
    --config "$CONFIG_PATH" \
    --remark "$REMARK" \
    --exp_name "$EXP_NAME" \
    --resume_step "$RESUME_STEP" \
    --if_rollout "$IF_ROLLOUT" \
    --roll_step "$ROLL_STEP"