#!/bin/bash

# configs=(
#   "beta=0.01 entropy_coef=0.01"
#   "beta=0.03 entropy_coef=0.01"
#   "beta=0.05 entropy_coef=0.01"
#   "beta=0.07 entropy_coef=0.01"
#   "beta=0.1 entropy_coef=0.01"
#   "beta=0.02 entropy_coef=0.02"
#   "beta=0 entropy_coef=0.01"
#   "beta=0 entropy_coef=0.02"
#   "beta=0 entropy_coef=0.03"
#   "beta=0 entropy_coef=0.04"
#   "beta=0 entropy_coef=0"
# )
configs=(
  "beta=0.07 entropy_coef=0.01 ent_coef_decay=False"

)

max_parallel=10
count=0

for config in "${configs[@]}"; do
  eval $config
  for seed in {1..20}; do
    echo "Running: seed=$seed, beta=$beta, entropy=$entropy_coef"
    PYTHONPATH=./torch-ac python3 rl-starter-files/rl-starter-files/scripts/train.py \
      --algo a2c \
      --seed $seed \
      --entropy-coef $entropy_coef \
      --ent_coef_decay $ent_coef_decay \
      --beta $beta &
    ((count++))
    if (( count % max_parallel == 0 )); then
      wait
    fi
  done
done

wait

