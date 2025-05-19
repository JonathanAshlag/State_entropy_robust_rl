import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib.ticker import LogLocator


# Load and filter
df = pd.read_csv("figs/beta_ablation.csv")
# df = df[df['wall'] == True]

# Group by agent and compute mean/std of success
group_stats = df.groupby(['agent', 'wall'])['success'].agg(['mean', 'std']).unstack()
agent_names = group_stats.index.tolist()

names= {
    "beta:80.0_alpha:0.01": 80,
    "beta:120.0_alpha:0.01": 120,
    "beta:160.0_alpha:0.01": 160,
    "beta:200.0_alpha:0.01": 200,
    "beta:240.0_alpha:0.01": 240,
    # "beta:0.0_alpha:0.02": "policy_entropy",
}


group_stats = df.groupby(['agent', 'wall'])['success'].agg(['mean', 'std']).unstack()
group_stats = group_stats.rename(index=names)      # map strings → ints
group_stats = group_stats.sort_index()   
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

colors = ['green', 'red']

fig, ax = plt.subplots(figsize=(10, 8))
rollous=[]
means_nominal=[]
stds_nominal=[]
means_purturbed=[]
stds_purturbed=[]
for agent in group_stats.index:
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
plt.plot(rollous,means_nominal,marker='o',color='#B07AA1' ,linewidth=3,markersize=5,label="nominal environment")
plt.fill_between(rollous, means_nominal - stds_nominal_smoove, means_nominal + stds_nominal_smoove,color='#B07AA1'  ,alpha=0.3)
plt.plot(rollous,means_purturbed,marker='o',color='#8C564B' ,linewidth=3,markersize=5,label="purturbed environment")
plt.fill_between(rollous, means_purturbed - stds_purturbed_smoove, means_purturbed + stds_purturbed_smoove,color='#8C564B'  ,alpha=0.3)
plt.axvline(160, color='grey', linestyle='--', linewidth=2, label="selected value")

# Formatting
# 
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylabel("Average success rate", fontsize=28)
plt.xlabel("state entropy regularization coefficient", fontsize=28)
plt.legend(fontsize=24,loc='best')
plt.title("Varying Regularization Strengths", fontsize=28)
plt.tight_layout()
plt.savefig("figs/ablation_beta.pdf")