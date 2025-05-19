import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import numpy as np
import utils
import matplotlib
from utils.custom_env2 import EmptyEnv
from utils.custom_env import CrossRoads
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
red_cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", "red"])
custom_cmap = ListedColormap(["white", "black"])
# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env",default='MiniGrid-EmptyEnv-v0',help="name of the environment (REQUIRED)")
parser.add_argument("--model", default='MiniGrid-EmptyEnv-v0_seed9_beta_0.07alpha_0.01',help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=50,help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=2,help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=1,help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,help="add a GRU to the model")
args = parser.parse_args()

# Set seed for all randomness sourcess

utils.seed(args.seed)

# Set device
models_dirs={"state_entropy":"MiniGrid-EmptyEnv-v0_seed9_beta_0.07alpha_0.01",
             "policy_entropy":"MiniGrid-EmptyEnv-v0_seed9_beta_0.0alpha_0.01",
             "unregularized":"MiniGrid-EmptyEnv-v0_seed9_beta_0.0alpha_0.0"}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")
wall = False
model_dir = utils.get_model_dir(args.model)
env = utils.make_env(args.env, blocked=wall )
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, num_envs=args.procs,
                    use_memory=args.memory, use_text=args.text)


print("Agent loaded\n")
# Load environments

trajectories = {"state_entropy":[], "policy_entropy":[],"unregularized":[]}
# Load agent
#run the agent on the environment:

for regularization in models_dirs.keys():
    model_dir=utils.get_model_dir(models_dirs[regularization])
    agent = utils.Agent(env.observation_space, env.action_space, model_dir, device=device, argmax=args.argmax, num_envs=args.procs,
                    use_memory=args.memory, use_text=args.text)
    for i in range(args.episodes):
        xy_s = []
        obs = env.reset()
        xy=np.array(env.agent_pos)
        xy[1] = 9-xy[1]
        done = False
        while not done:
            xy_s.append(xy)
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action) 
            xy = np.array(env.agent_pos)
            xy[1] = 9-xy[1]
        
        xy = np.array(env.agent_pos)
        xy[1] = 9-xy[1]
        xy_s.append(xy)

        
        trajectories[regularization].append(np.array(xy_s))

    #plot the trajectories:
colors = {"state_entropy":"green", "policy_entropy":"red","unregularized":"blue"}

for regularization in models_dirs.keys():
    for trajectory in trajectories[regularization]:
        epsilon=np.random.randn(2)/8
        plt.plot(trajectory[:,0]+epsilon[0], trajectory[:,1]+epsilon[1], color=colors[regularization], alpha=0.3,label=regularization)

if wall: 
    wall_line=np.array([(4,4),(4,5),(4,6),(4,7),(4,8)])
    plt.plot(wall_line[:,0],wall_line[:,1], color='black', linewidth=10, label="wall")

plt.scatter(1,8, color='yellow', marker='*', s=100, label="goal")
plt.title("Loacal Catastrophie")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("wall_Environment.png")
    


        

    #compute the trajectories2 std:
