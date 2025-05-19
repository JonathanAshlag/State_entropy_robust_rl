import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from matplotlib.ticker import LogLocator

df = pd.read_csv("figs/distill_policy_entropy.csv")
# policy_entropy_df = pd.read_csv("figs/distill_policy_entropy.csv")
# policy_policy_entropy_df = pd.read_csv("figs/distill_policy_entropy.csv")
# policy_entropy_df = pd.DataFrame('figs/distill_policy_entropy.csv')
#plot y axis="wandb_group_name: action_entropy_0.0__policy_entropy_160.0_160.0 - charts/episodic_policy_entropy" , x axis global_step:
# Create the plot
metric_cols = [
    c for c in df.columns 
    if c.endswith('losses/entropy') and '__' not in c
]


mean_raw = df[metric_cols].mean(axis=1)
std_raw  = df[metric_cols].std(axis=1)

# 4. (optional) Gaussian‐smooth both series
sigma = 2  # tweak for more or less smoothing
mean_s = gaussian_filter1d(mean_raw, sigma)
std_s  = gaussian_filter1d(std_raw,  sigma)

# 5. plot
x = df['global_step']
# plt.rcParams.update({
#     'font.size': 24,
#     'axes.titlesize': 26,
#     'axes.labelsize': 24,
#     'legend.fontsize': 26,
#     'legend.title_fontsize': 26,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20
# })
plt.figure()
plt.plot(x, mean_s, label='mean ')
plt.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.3, label=' std ')
plt.xlabel('global_step')
plt.ylabel('policy_entropy')
plt.title('Mean ± STD of state entropy through distillation')
# plt.yticks(np.arange(0, 0.1, 0.01))
# plt.tight_layout()
plt.savefig('figs/distill_policy_entropy.png')