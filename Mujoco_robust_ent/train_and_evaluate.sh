
configs=(
  "beta=160.0 starting_beta=800.0 ent_coef=0.01 save_dir=pusher_puck"
)
max_parallel=1
count=0
# 
for seed in 1; do
  for config in "${configs[@]}"; do
    eval $config
    echo "Running: seed=$seed, num_envs $num_envs"
    python3 lstm_continuous_action_ppo.py \
      --seed $seed \
      --ent_coef $ent_coef \
      --starting_beta $starting_beta \
      --save_dir $save_dir \
      --beta $beta &
    ((count++))
    if (( count % max_parallel == 0 )); then
      wait
    fi
  done
done
python3 Eval_pusher_wall.py \
  --seed $seed \
  --ent_coef $ent_coef \
  --starting_beta $starting_beta \
  --load_dir $save_dir \
  --beta $beta
wait
