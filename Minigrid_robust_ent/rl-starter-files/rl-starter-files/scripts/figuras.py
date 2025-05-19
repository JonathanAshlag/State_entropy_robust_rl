import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import sem

#False_False_beta0_entropy0.01.npy
baselines=["True_False_beta0_entropy0.01","True_False_beta0_entropy0.02","True_False_beta0.05_entropy0.01"]
labels ={"True_False_beta0_entropy0.01": "action only, entropy-coef=0.01","True_False_beta0_entropy0.02": "action only, entropy-coef=0.02","True_False_beta0.05_entropy0.01": "SE+action,beta=0.05 entropy-coef=0.01"}
colors ={"True_False_beta0_entropy0.01": "blue","True_False_beta0_entropy0.02": "red","True_False_beta0.05_entropy0.01": "green"}

data ={header: {} for header in baselines}
for header in baselines:
    for i in range(1,6):
        data[header][i] = np.load('/home/yonatanashlag/RobustEnt/VCSE/VCSE_A2C/rl-starter-files/rl-starter-files/scripts/Evaluations2/MiniGrid-EmptyEnv-v0_seed'+str(i)+'_'+header+'.npy')
        
    #padd the data, so that all the arrays have the same length, to the samller arrays just add the last value of the array:
    max_len = 1000
    for j in range(1,6):
        data[header][j] = np.pad(data[header][j], (0,max_len-len(data[header][j])), 'edge')
    
    #stack the arrays:
    data[header] = np.stack([data[header][i] for i in range(1,6)], axis=0)
    time = np.arange(0,100)
    mean = np.mean(data[header], axis=0)[:100]
    stderr =np.std(data[header], axis=0)[:100]
    ci =  stderr
    label = labels[header]
    color = colors[header]
    plt.plot(time, mean,linestyle="-",color=color ,label=label)
    #plot confidence interval:
    plt.fill_between(time, mean-ci, mean+ci, color=color, alpha=0.3)
#add vertical line at 15:
plt.axvline(x=15, color='black', linestyle='--',label='Vannilla A2C')


plt.xlabel('Test Horizon')
plt.ylabel('Success Rate')
plt.legend()
plt.title('Success Rate vs Test Horizon')
plt.savefig(fname = 'VCSE_A2C/rl-starter-files/rl-starter-files/scripts/figs2/EmptyEnv.png')



