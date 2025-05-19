import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
#load the data
#palette = {
#     'CMD-1': '#B07AA1',   # soft lavender-pink  
#     'CMD-2': '#6A0572',   # deep purple  
#     'CRL' :  '#EDC948',   # mustard yellow  
#     'QRL' :  '#F28E2B',   # vivid orange  
#     'DDPG':  '#9C755F',   # medium brown  
#     'GCBC':  '#8C564B',   # dark chocolate brown  
# }

#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your dataframe
df = pd.read_csv("figs/pusher_wall_performance_final_0.csv")
# df =pd.read_csv("figs/state_or_entropy.csv")
#applay absolute value to distance;
df['distance'] = df['distance'].abs()
# Group by agent and wall, calculate mean and std over seeds
group_stats = df.groupby(['agent', 'wall'])['success'].agg(['mean', 'std']).unstack()

# Bar plot with error bars
agents = group_stats.columns.levels[0]  # ["mean", "std"]
walls = [False, True]
agent_names = group_stats.index.tolist()
num_agents = len(agent_names)
x = np.arange(2)
width = 0.35
x_names=["nominal environment","pertubed environment"]

fig, ax = plt.subplots(figsize=(10, 8))
labels ={"beta:0.0_alpha:0.02": "policy entropy","beta:160.0_alpha:0.01":"state entropy","beta:0.0_alpha:0.0":"unregularized"}
# labels = {"beta:160.0_alpha:0.01":"state & policy entropy","beta:160.0_alpha:0.0":"state entropy","beta:160.0_alpha:0.0_decayd":"state entropy alpha decay",}
colors = ['#EDC948', '#F28E2B', '#6A0572']
for wall in [False, True]:
    if wall:
        index=1
    else:
        index=0
    for i, agent in enumerate(agent_names):
        if agent in labels.keys():
            color = colors[i]
            means = group_stats.loc[agent, ('mean', wall)]
            stds = group_stats.loc[agent, ('std', wall)]
            
            # Plot no-wall (left bar) and wall (right bar) with same base color
            if index==0:
                ax.bar(x[index] - (1-i)*width/2, means, width=width/3, yerr=1.96/5*stds, label=labels[agent], color=colors[i], alpha=0.7, capsize=5)
            else:
                ax.bar(x[index] - (1-i)*width/2, means, width=width/3, yerr=1.96/5*stds, color=colors[i], alpha=0.7, capsize=5)
        # ax.bar(x[i] + width/2, means[1], width=width/2, yerr=1.96/20*stds[1], color=colors[], alpha=1.0, capsize=5)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(x_names, fontsize=28)
ax.set_ylabel("Average success rate", fontsize=28)
plt.yticks(size=24)
plt.ylim(0, 1.05)  
ax.set_title("Regularizations comparison", fontsize=28)
ax.legend(fontsize=24)
plt.tight_layout()
plt.savefig("figs/pusher_wall_bar.pdf")
# plt.savefig("figs/state_or_entropy_bar.png")