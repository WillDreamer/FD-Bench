#!/bin/bash


# ------------------------
# 可选范围
# ------------------------
VALID_SPATIAL=("self_atten" "graph" "latent" "fourier" "conv" "diffusion" "flow")
VALID_TEMPORAL=("self_atten" "next_step" "temporal_bundling" "auto_regressive" "node")
VALID_TARGET=("variable" "noise" "flow")

# 默认值
SPATIAL_REP="fourier"
TEMPORAL_REP="next_step"
TARGET="variable"
REMARK="default_run"

print_help() {
    echo "Usage: bash train.sh --spatial <SPATIAL> --temporal <TEMPORAL> --target <TARGET> --remark <REMARK>"
    echo ""
    echo "Options:"
    echo "  --spatial   : ${VALID_SPATIAL[*]}"
    echo "  --temporal  : ${VALID_TEMPORAL[*]}"
    echo "  --target    : ${VALID_TARGET[*]}"
    echo "  --remark    : 自定义备注字符串"
    echo ""
    echo "Example:"
    echo "  bash train.sh --spatial fourier --temporal next_step --target variable --remark test_run"
    exit 1
}

check_valid() {
    local val=$1
    shift
    local arr=("$@")
    for item in "${arr[@]}"; do
        if [[ "$val" == "$item" ]]; then
            return 0
        fi
    done
    return 1
}

# ------------------------
# 参数解析
# ------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --spatial)
      SPATIAL_REP=$2
      shift 2
      ;;
    --temporal)
      TEMPORAL_REP=$2
      shift 2
      ;;
    --target)
      TARGET=$2
      shift 2
      ;;
    --remark)
      REMARK=$2
      shift 2
      ;;
    --help|-h)
      print_help
      ;;
    *)
      echo "Unknown argument: $1"
      print_help
      ;;
  esac
done

# ------------------------
# 参数合法性检查
# ------------------------
if ! check_valid "$SPATIAL_REP" "${VALID_SPATIAL[@]}"; then
    echo "Error: Invalid spatial option '$SPATIAL_REP'"
    echo "Valid options: ${VALID_SPATIAL[*]}"
    exit 1
fi

if ! check_valid "$TEMPORAL_REP" "${VALID_TEMPORAL[@]}"; then
    echo "Error: Invalid temporal option '$TEMPORAL_REP'"
    echo "Valid options: ${VALID_TEMPORAL[*]}"
    exit 1
fi

if ! check_valid "$TARGET" "${VALID_TARGET[@]}"; then
    echo "Error: Invalid target option '$TARGET'"
    echo "Valid options: ${VALID_TARGET[*]}"
    exit 1
fi

# ------------------------
# 运行训练
# ------------------------
CONFIG_PATH="config/${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: The modular combination ${TARGET}/${SPATIAL_REP}+${TEMPORAL_REP}.yaml does not exist!"
    exit 1
fi

export WANDB_ENTITY="FD-Bench"
export WANDB_PROJECT="${WANDB_ENTITY}_${TARGET}"
export WANDB_NAME="${SPATIAL_REP}_${TEMPORAL_REP}_${REMARK}"
export WANDB_API_KEY="xxxxxxxxxxx"

accelerate launch --main_process_port 29514 src/train.py \
    --config_file "$CONFIG_PATH" \
    --remark "$REMARK"
