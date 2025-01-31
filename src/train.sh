# FD-Bench/train.sh
#!/bin/bash


# SPATIAL_REP="self_atten"
SPATIAL_REP="fourier"
TEMPORAL_REP="next_step"
TARGET="var"
CONFIG_PATH="config/${SPATIAL_REP}+${TEMPORAL_REP}+${TARGET}.yaml"


# 运行训练
python src/train.py --config "$CONFIG_PATH"