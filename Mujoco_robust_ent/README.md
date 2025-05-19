

# State Entropy Regularization For Robust RL

This repository is the official implementation of [State Entropy Regularization For Robust RL], subbmited to Neurips 2025.

## Requirements

To install requirements:

```setup
conda env create -f env.yml
```

## Training

lstm_continuous_action_ppo.py is the training script:

ent_coef - Policy entropy coefficient 

starting_beta - Initial state entropy coefficient

beta - Final state entropy coeficient

beta_decay -state entropy intrinsic reward discount factor

ent_coef_decay -whether to linearly decay the policy entropy to 0 throught training

num_envs - the number of parallel game environments dictates the number of roolouts per update, state entropy estimation


## Evaluation

The wall expirement is Eval_pusher_wall. To train and evaluate a policy run:

```eval
bash train_and_evaluate.sh
```
hyperparameters can be specified in the configs variable inside.
the evaluation results will be saved in a .csv file

 
