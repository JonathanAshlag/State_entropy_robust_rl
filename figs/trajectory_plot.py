# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

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
from lstm_continuous_action_puck import Agent
from tqdm import tqdm
import pandas as pd
import mujoco_local
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# option A: keep the alias you’re already using
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as patches

def parse_args():
    parser = argparse.ArgumentParser()

    # Algorithm specific argumentsObstaclePusher-v0#we did 100 eval steps
    parser.add_argument("--exp_name", default=os.path.basename(__file__)[: -len(".py")], help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=10, help="seed of the experiment")
    parser.add_argument("--torch_deterministic", type=bool, default=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=bool, default=False, help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb_project_name", default="RobustEnt_Mujoco", help="the wandb's project name")
    parser.add_argument("--wandb_entity", default="yonatan-ashlag1", help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=bool, default=False, help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--eval_steps", type=int, default=100, help="the number of steps to run in each environment per update")
    parser.add_argument("--env_id", default="CustomPusher-v1", help="the id of the environment")
    parser.add_argument("--eval_episodes", type=int, default=50, help="the number of evaluation episodes")
    parser.add_argument("--num_envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--ent_coef", type=float, default=0.001, help="coefficient of the entropy")
    parser.add_argument("--beta", type=float, default=0.03, help="coefficient of the state entropy")
    parser.add_argument("--network_hidden_size", type=int, default=256, help="the size of the hidden layer in the network")
    # to be filled in runtime

    args = parser.parse_args()
    args.wandb_group_name = f"action_entropy_{args.ent_coef}__state entropy_{args.beta}_fixed_reset"
    return args



def make_env(env_id, idx, capture_video, run_name, gamma, control_cost=0,horizon=100,xml_file=None,success_truncation=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id,reward_control_weight=control_cost, render_mode="rgb_array",xml_file=xml_file,success_truncation=success_truncation)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id,reward_control_weight=control_cost,xml_file=xml_file,success_truncation=False)
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
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=False,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # ent_coefs=[0.0, 0.001, 0.01]
    # betas=[0.0, 0.005, 0.01, 0.05]
    # configs = []
    
    # for ent_coef in ent_coefs:
    #     for beta in betas:
    #         configs.append({"ent_coef": ent_coef, "beta": beta})
    configs = [{"ent_coef": 0.00, "beta": 0.0,"seed":21,"name":"unregularized"},
               {"ent_coef": 0.02, "beta": 0.0,"seed":3,"name":"policy entropy"},#2 for no wall
               {"ent_coef": 0.01, "beta": 160.0,"seed":11,"name":"state entropy"}]#before

    

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # joints=["r_shoulder_pan_joint",
    #         "r_shoulder_lift_joint",
    #         "r_upper_arm_roll_joint",
    #         "r_elbow_flex_joint",#
    #         "r_forearm_roll_joint",
    #         "r_wrist_flex_joint",
    #         "r_wrist_roll_joint",]
    
    args.reapets = args.eval_episodes // args.num_envs
    trajectories ={"unregularized":[], "policy entropy":[], "state entropy":[]}
    wall_trajectories ={"unregularized":[], "policy entropy":[], "state entropy":[]}
    min_reward = 100
    all_rewards=[]
    for wall in [False,True]: #[0.1,0.3,0.5]     
        if wall:
            xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/custom_pusher_red_obstacle.xml")
        else:
            xml_file = os.path.join(os.path.dirname(__file__), "mujoco_local/pusher_v5.xml")

        envs = gym.vector.SyncVectorEnv(
                        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma, xml_file=xml_file,horizon=args.eval_steps,success_truncation=True) for i in range(args.num_envs)]
                    )
        for seed in [21]:#2,
            for i, conf in tqdm(enumerate(configs)):
                    agent = Agent(envs).to(device)
                    path = f"runs_puck_final/CustomPusher-v1__{conf['seed']}__alpha_{conf['ent_coef']}_beta_{conf['beta']}_grad_slow"
                    agent.load_state_dict(torch.load(f"{path}/agent.pth", weights_only=True))
                    normalize = np.load(f"{path}/agent_normalize.npz", allow_pickle=True)
                    norm_mean, norm_var = normalize["normalize_mean"].mean(), normalize["normalize_var"].mean()
                    k=0
                    while k <args.repeats:
                        trajectory =[]
                        obss, _ = envs.reset(seed=args.seed+k)
                        obss = torch.Tensor(normalize_obs(obss, norm_mean, norm_var)).to(device)
                        next_lstm_state = (
                            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
                        ) 
                        dones = torch.zeros(args.num_envs, device=device)
                        step = 0

                        mask = torch.ones(args.num_envs, device=device)
                        while step < args.eval_steps:
                            with torch.no_grad():
                                actions, logprob, _, value, next_lstm_state = agent.get_action_and_value(obss, next_lstm_state, dones)

                            obss, rewards, dones, truncations, infos = envs.step(actions.cpu().numpy())
                            puck_xy = obss[0][17:19]
                            trajectory.append(puck_xy)
                            obss = torch.Tensor(normalize_obs(obss, norm_mean, norm_var)).to(device)
                            dones = torch.Tensor(dones).to(device)

                            step += 1
                            if dones or truncations:
                                final_dist = infos["final_info"][0]["reward_dist"]
                                if final_dist >-0.4:
                                    k+=1
                                reward =infos["final_info"][0]["episode"]["r"]
                                all_rewards.append(reward)
                                if reward < min_reward:
                                    min_reward = reward
                                break
                        
                        # Save the trajectory
                        
                        # Save the trajectory to the dictionary
                        if final_dist >-0.4:
                            if wall:
                                wall_trajectories[conf["name"]].append((np.array(trajectory[:-1]),reward.item()))
                            else:
                                trajectories[conf["name"]].append((np.array(trajectory[:-1]),reward.item()))


# all_rewards = np.array(all_rewards)
# max_reward = all_rewards.max()
# all_rewards /= max_reward
# std= np.std(all_rewards).item()
                            
colors = ['#EDC948', '#F28E2B', '#6A0572']    
alphas = [0.6, 0.3, 0.1]    
fig, axes = plt.subplots(1,2, figsize=(12,5), sharex=True, sharey=True)
titles = ["Nominal Environment", "Local Perturbation"]

for ax, trajs,title in zip(axes, [trajectories, wall_trajectories], titles):
    for i,key in enumerate(trajs.keys()):
        for j,traj in enumerate(trajs[key]):
            # alpha = np.clip(traj[1]/max_reward,0.01,0.99).item()
            ax.plot(traj[0][:-1, 0], traj[0][:-1, 1], color=colors[i], alpha=alphas[i], label=key if j == 0 else "")

        ax.scatter(0.45, -0.05, color='green', marker='*', label='goal', s=100,zorder=10)

        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False)
        ax.set_title(title, fontsize=20)

for x, y in [(0.55, -0.31), (0.58, -0.28), (0.61, -0.25)]:
    obstacle = patches.Rectangle(
        (x, y),        # bottom-left corner
        0.03, 0.03,   # size of the rectangle
        linewidth=0,
        edgecolor=None,
        facecolor='red',
        alpha=0.8,
        zorder=20         # behind the trajectories
    )
    axes[1].add_patch(obstacle)

legend_handles = [
    Line2D([0],[0], color='#EDC948', lw=3, label='Standard RL'),
    Line2D([0],[0], color='#F28E2B', lw=3, label='Policy Entropy Regularization'),
    Line2D([0],[0], color='#6A0572', lw=3, label='State Entropy Regularization'),
    Line2D([0],[0], marker='*', color='w', markerfacecolor='green',
           markersize=15, label='goal')
]
fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=len(legend_handles),
    frameon=False,
    bbox_to_anchor=(0.5, 0.0),#-0.05
    fontsize=14
)
plt.subplots_adjust(bottom=0.20)
fig.tight_layout()
fig.savefig("figs/trajectories.pdf",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.1)  
