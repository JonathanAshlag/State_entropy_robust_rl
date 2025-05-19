import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib.ticker import LogLocator


# Load and filter
df = pd.read_csv("figs/wall_ablations.csv")
# df = df[df['wall'] == True]

# Group by agent and compute mean/std of success
group_stats = df.groupby(['agent', 'wall'])['success'].agg(['mean', 'std']).unstack()
agent_names = group_stats.index.tolist()

names= {
    "beta:30.0_alpha:0.01_4_rollouts": 4,
    "beta:60.0_alpha:0.01_8_rollouts": 8,
    "beta:100.0_alpha:0.01_16_rollouts": 16,
    "beta:160.0_alpha:0.01_32_rollouts": 32,
    # "beta:0.0_alpha:0.02": "policy_entropy",
}


group_stats = df.groupby(['agent', 'wall'])['success'].agg(['mean', 'std']).unstack()
#replace agent names with numbers:
for i, agent in enumerate(agent_names):
    if agent in names:
        group_stats.rename(index={agent: names[agent]}, inplace=True)
    else:
        group_stats.rename(index={agent: i}, inplace=True)
# after renaming
agent_names = list(group_stats.index)

agents = group_stats.columns.levels[0]  # ["mean", "std"]
walls = [False, True]
wall_preformances ={}

for wall in [False, True]:
    if wall:
        index=1
    else:
        index=0
    for i, agent in enumerate(agent_names):
        means = group_stats.loc[agent, ('mean', wall)]
        # Store wall performance for later
        wall_preformances[agent] = means
        
fig, ax = plt.subplots(figsize=(10, 8))

sorted_agents = sorted(wall_preformances.items(), key=lambda x: x[1], reverse=True)
sorted_agents = [x[0] for x in sorted_agents]
rollous=[]
means_nominal=[]
stds_nominal=[]
means_purturbed=[]
stds_purturbed=[]
for i, agent in enumerate(sorted_agents):
    rollous.append(agent)
    means_p = group_stats.loc[agent, ('mean', True)]
    stds_p = 1.96/np.sqrt(25) *group_stats.loc[agent, ('std', True)]
    means_n = group_stats.loc[agent, ('mean', False)]
    stds_n = 1.96/np.sqrt(25) *group_stats.loc[agent, ('std', False)]

    means_nominal.append(means_n)
    stds_nominal.append(stds_n)
    means_purturbed.append(means_p)
    stds_purturbed.append(stds_p)
        
rollous=np.array(rollous)
means_nominal=np.array(means_nominal)
stds_nominal=np.array(stds_nominal)
means_purturbed=np.array(means_purturbed)
stds_purturbed=np.array(stds_purturbed)

stds_nominal_smoove=gaussian_filter1d(stds_nominal, sigma=1)
stds_purturbed_smoove=gaussian_filter1d(stds_purturbed, sigma=1)
#plot 2 lines with smooved guassian std#8C564B
plt.plot(rollous,means_nominal,marker='o',color='#B07AA1',linewidth=3 ,markersize=5,label="nominal environment")
plt.fill_between(rollous, means_nominal - stds_nominal_smoove, means_nominal + stds_nominal_smoove,color='#B07AA1'  ,alpha=0.3)
plt.plot(rollous,means_purturbed,marker='o',color='#8C564B',linewidth=3 ,markersize=5,label="purturbed environment")
plt.fill_between(rollous, means_purturbed - stds_purturbed_smoove, means_purturbed + stds_purturbed_smoove,color='#8C564B'  ,alpha=0.3)
plt.axhline(y=0.029750000443309537, color='grey',linewidth=2, linestyle='--', label='policy entropy')
# Formatting
#add (4,3) figsize:
plt.xticks(size=24)
plt.yticks(size=24)
plt.ylabel("Average success rate", fontsize=28)
plt.xscale('log', base=2)
plt.xlabel("number of rollouts per entropy estimation", fontsize=28)
plt.title("Rollouts Per Entropy Etimation Ablation", fontsize=28)
plt.legend(fontsize=24)
# plt.rcParams.update({
#     'font.size': 24,
#     'axes.titlesize': 26,
#     'axes.labelsize': 24,
#     'legend.fontsize': 26,
#     'legend.title_fontsize': 26,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20
# })
plt.tight_layout()
plt.savefig("figs/pusher_rollout_sensitivity.pdf")
# plt.plot(xs, ys, marker='o', markersize=5)

# # Plot smoothed CI band
# plt.fill_between(xs, ys - yerrs_smooth, ys + yerrs_smooth, alpha=0.3, label='Smoothed 95% CI')
