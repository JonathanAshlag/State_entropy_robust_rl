# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

import os
import random
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from normalize import NormalizeObservation
import utils
import argparse
from lstm_continuous_action_ppo import Agent
from tqdm import tqdm
import pandas as pd
import mujoco_local
import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm specific argumentsObstaclePusher-v0
    #fill the arguments of the policy you wish to evaluate
    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=10, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--eval_steps", type=int, default=100, help="the number of steps to run in each environment per update")
    parser.add_argument("--env_id", default="CustomPusher-v1", help="the id of the environment")
    parser.add_argument("--eval_episodes", type=int, default=200, help="the number of evaluation episodes")
    parser.add_argument("--num_envs", type=int, default=20, help="the number of parallel game environments")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--beta", type=float, default=160.0, help="coefficient of the state entropy")
    parser.add_argument("--starting_beta", type=float, default=800.0, help="coefficient of the state entropy warmup")
    parser.add_argument("--test_seeds", type=list, default=[1], help="the evaluation seeds")
    parser.add_argument("--network_hidden_size", type=int, default=256, help="the size of the hidden layer in the network")
    parser.add_argument("--load_dir", type=str, default="runs_puck_final", help="The trained agent's directory")
    # to be filled in runtime

    args = parser.parse_args()
    args.wandb_group_name = f"action_entropy_{args.ent_coef}__state_entropy_{args.beta}_fixed_reset"
    return args



def make_env(env_id, idx, run_name, gamma, control_cost=0,horizon=100,xml_file=None,success_truncation=False):
    def thunk():
        env = gym.make(env_id,reward_control_weight=control_cost,xml_file=xml_file,success_truncation=success_truncation)
        env=gym.wrappers.TimeLimit(env, max_episode_steps=horizon)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
       
        
        return env

    return thunk

def normalize_obs(obs, mean, var):
    """Normalizes the observation using the running mean and variance of the observations."""
    epsilon = 1e-8  # small constant to avoid division by zero
    return (obs - mean) / np.sqrt(var + epsilon)

if __name__ == "__main__":
    args = parse_args()
    args.repeats = args.eval_episodes // args.num_envs
    run_name = f"{args.env_id}__{args.seed}__alpha_{args.ent_coef}__beta_{args.beta}"
 
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    configs = [{"ent_coef": args.ent_coef, "beta": args.beta,"starting_beta": args.starting_beta}]
   
    
    df = pd.DataFrame(columns=['agent','seed' ,'reward','distance','wall','success'])

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.reapets = args.eval_episodes // args.num_envs
  
    for wall in [True,False]: #[0.1,0.3,0.5]     
        if wall:
            xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/custom_pusher_red_obstacle.xml")
        else:
            xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/pusher_v5.xml")

        envs = gym.vector.SyncVectorEnv(
                        [make_env(args.env_id, i, run_name, args.gamma, xml_file=xml_file,horizon=args.eval_steps,success_truncation=True) for i in range(args.num_envs)]
                    )
        for seed in args.test_seeds:#2,
            for i, conf in tqdm(enumerate(configs)):
                    agent = Agent(envs).to(device)
                    path = f"{args.load_dir}/CustomPusher-v1__{seed}__alpha_{conf['ent_coef']}_beta_{conf['beta']}_starting_beta_{conf['starting_beta']}"
                    agent.load_state_dict(torch.load(f"{path}/agent.pth", weights_only=True))
                    normalize = np.load(f"{path}/agent_normalize.npz", allow_pickle=True)
                    norm_mean, norm_var = normalize["normalize_mean"].mean(), normalize["normalize_var"].mean()

                    episode_rewards = []
                    episode_distances = []
                    episode_successes = []
                    for k in range(args.repeats):
                        obss, _ = envs.reset(seed=args.seed+k)
                        obss = torch.Tensor(normalize_obs(obss, norm_mean, norm_var)).to(device)
                        next_lstm_state = (
                            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                        ) 
                        dones = torch.zeros(args.num_envs, device=device)
                        step = 0
                        cumulative_rewards = torch.zeros(args.num_envs, device=device)
                        final_distances = torch.zeros(args.num_envs, device=device)
                        successes = torch.zeros(args.num_envs, device=device)
                        mask = torch.ones(args.num_envs, device=device)
                        while step < args.eval_steps:
                            with torch.no_grad():
                                actions, logprob, _, value, next_lstm_state = agent.get_action_and_value(obss, next_lstm_state, dones)

                            obss, rewards, dones, truncations, infos = envs.step(actions.cpu().numpy())
                            obss = torch.Tensor(normalize_obs(obss, norm_mean, norm_var)).to(device)
                            dones = torch.Tensor(dones).to(device)
                            rewards = torch.Tensor(rewards).to(device)

                            cumulative_rewards += rewards * mask

                            for idx, done in enumerate(dones):
                                if done or truncations[idx]:
                                    mask[idx] = 0
                                    if 'final_info' in infos:
                                        final_distances[idx] = infos['final_info'][idx]['reward_dist']
                                        successes[idx] = infos['final_info'][idx]['success']
                                    else:
                                        final_distances[idx] = infos['reward_dist'][idx]
                                        successes[idx] = infos['success'][idx]
                                

                            step += 1

                        episode_rewards.append(cumulative_rewards.mean().item())
                        episode_distances.append(final_distances.mean().item())
                        episode_successes.append(successes.mean().item())


                    avg_reward = np.mean(episode_rewards)
                    avg_distance = np.mean(episode_distances)
                    avg_success = np.mean(episode_successes)
                    df.loc[len(df)] = [f"beta:{conf['beta']}_alpha:{conf['ent_coef']}", seed, avg_reward, avg_distance,wall,avg_success]


    df.to_csv('pusher_wall_alpha.csv', index=False)
   
    #print average success rate for wall and no wall:
    wall_success = df[df['wall'] == True]['success'].mean()
    no_wall_success = df[df['wall'] == False]['success'].mean()
    print(f"Agent:alpha_{conf['ent_coef']}_beta_{conf['beta']}_starting_beta_{conf['starting_beta']}")
    print(f"Average success rate with wall: {wall_success}")
    print(f"Average success rate nominal Environment: {no_wall_success}")