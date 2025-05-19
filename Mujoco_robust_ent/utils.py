import torch
import numpy as np
import torch
import numpy as np
import time
import torch
import random
import torch.nn.functional as F
import torch.distributions as torchd
def compute_state_entropy(src_feats, tgt_feats, average_entropy=False, k_fixed=5,batch=False,calc_steps=1):
    # src_feats: (num_steps,num_envs, feat_dim) -> (num_envs*num_steps, feat_dim)
    # tgt_feats: (num_steps,num_envs, feat_dim) -> (num_envs*num_steps, feat
    if not batch:
        num_steps, feat_dim =  tgt_feats.size()#num|_envs
    else:
        num_steps,num_envs, feat_dim =  tgt_feats.size()#num|_envs
    step_size = num_steps//calc_steps
    state_entropies= torch.zeros(num_steps)
    with torch.no_grad():
        src_feats = src_feats.reshape(-1, feat_dim)
        tgt_feats = tgt_feats.reshape(-1, feat_dim)
        for i in range(calc_steps):
            src_feats_i = src_feats[i*step_size:(i+1)*step_size]
            dist = torch.norm(
                src_feats_i[:, None, :] - tgt_feats[ None, :, :], dim=-1, p=2
            )
            
            knn_dists = 0.0
            if average_entropy:
                for k in range(k_fixed):
                    knn_dists += torch.kthvalue(dist, k + 1, dim=1).values
                knn_dists /= k_fixed
            else:
                knn_dists = torch.kthvalue(dist, k=k_fixed, dim=1).values
            state_entropy = knn_dists
            #state_entropy: (num_envs*num_steps,)->(num_envs,num_steps) not 
            # state_entropy = state_entropy.reshape(step_size, -1)
            state_entropy = torch.log(state_entropy+1)#notice that we add 1 to avoid log(0)
            state_entropies[i*step_size:(i+1)*step_size] = state_entropy
    return state_entropies




def state_entropy_sanity_check(src_feats, tgt_feats, average_entropy=False, k_fixed=5):
    # src_feats: (num_steps,num_envs, feat_dim) -> (num_envs*num_steps, feat_dim)
    # tgt_feats: (num_steps,num_envs, feat_dim) -> (num_envs*num_steps, feat
    num_steps,num_envs, feat_dim = src_feats.size()
    tgt_feats = tgt_feats.view(-1, feat_dim)
    knn_dists = torch.zeros(num_steps, num_envs)
    for i in range(num_steps):
        for j in range(num_envs):

            feat = src_feats[i,j]
            #compute the distance between feat and all the tgt_feats
            dist = torch.norm(feat - tgt_feats, dim=-1, p=2)
            #sort the distances
            knn_dists[i,j] = torch.kthvalue(dist, k=k_fixed, dim=0).values
    return knn_dists


def fill_batch(starting_points, obs, actions, i, num_envs, batch_length):
    device = obs.device
    batch_obs = torch.zeros((num_envs, batch_length, obs.shape[-1])).to(device)
    batch_actions = torch.zeros((num_envs, batch_length,actions.shape[-1])).to(device)
    batch_firsts = torch.zeros((num_envs, batch_length)).to(device)
    if i ==0:
        batch_firsts[:,0] = 1
    for j in range(num_envs):
        batch_obs[j] = obs[j,starting_points[j]:starting_points[j]+batch_length]
        batch_actions[j] = actions[j,starting_points[j]:starting_points[j]+batch_length]
        

    return batch_obs, batch_actions, batch_firsts

            
def list_mean(list_input):
    return np.array(list_input).mean()

def get_epochs(current_iter,args):
    if current_iter<args.warm_up:
        return 0
    elif current_iter==args.warm_up:
        return args.initial_epochs
    else:
        return args.dreamer_epochs
    

def sample_traj(train_episodes,batch_size):
    #random sampling:
    traj_indices = random.sample(range(len(train_episodes)),batch_size)
    traj_obs = torch.stack([train_episodes[traj_index]['obs'] for traj_index in traj_indices])
    traj_actions = torch.stack([train_episodes[traj_index]['action'] for traj_index in traj_indices])
    return traj_obs, traj_actions




    
def step_decreasing(current_iter, start_value, horizon):
    #in total we have horizon//start_value possible values
    #we want to decrease the value by 1 every horizon//start_value steps
    #so we have the following formula:
    #current_iter//decreasing_step = horizon//start_value
    #decreasing_step = current_iter/(horizon//start_value)

    return max(3,(horizon-current_iter)//(horizon//(start_value-1)))#we want to make sure that the value is at least 3
    

def entropy_from_logits(logits,unimix_ratio):
    probs = F.softmax(logits, dim=-1)
    probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
    #get the joint probability of the full stoch transistion



    entropy_per_discreate = -torch.sum(probs * torch.log(probs), dim=-1)
    joint_sampling_entropy = torch.sum(entropy_per_discreate, dim=-1)
    return joint_sampling_entropy





def tensor_correlation(x: torch.Tensor, y: torch.Tensor):
    """
    Compute the correlation between two tensors of shape [N, D].

    Args:
        x (torch.Tensor): Tensor of shape [N, D]
        y (torch.Tensor): Tensor of shape [N, D]

    Returns:
        torch.Tensor: Correlation coefficients of shape [D]
    """
    # Ensure tensors are float
    x = x.float()
    y = y.float()
    #reshape the tensors (steps,envs,feat_dim)->(steps*envs,feat_dim)
    x = x.reshape(-1,x.size(-1))
    y = y.reshape(-1,y.size(-1))

    # Compute means
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)

    # Center the tensors
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Compute covariance
    cov_xy = (x_centered * y_centered).sum(dim=0) / (x.shape[0] - 1)

    # Compute standard deviations
    std_x = x_centered.pow(2).sum(dim=0).sqrt() / (x.shape[0] - 1) ** 0.5
    std_y = y_centered.pow(2).sum(dim=0).sqrt() / (y.shape[0] - 1) ** 0.5

    # Compute correlation
    correlation = cov_xy / (std_x * std_y + 1e-8)  # Add epsilon to prevent division by zero

    return correlation


def predicted_rollout(deters,stochs,dynamics_model,rollout_length,action_sequence,obs_dim):
    # latents: (num_envs,num_steps, latent_dim)
    # action_sequence: (rollout_length, action_dim)
    num_steps,num_envs, deter_dim = deters.size()
    _,_, stoch_dim = stochs.size()
    _,repeat,_,action_dim = action_sequence.size()
    predicted_rollouts = torch.zeros((num_steps,num_envs,repeat*rollout_length,obs_dim))
    rollout_std = torch.zeros((num_steps,num_envs)).to(deters.device)
    #preprocess the latents for the dynamics model
    deters = deters.reshape(-1,deter_dim)
    stochs = stochs.reshape(-1,stoch_dim)

    next_latents = {'deter':deters,'stoch':stochs}
    with torch.no_grad():
        for j in range(repeat):
            for i in range(rollout_length):
                next_latents = dynamics_model.dynamics.img_step(prev_state=next_latents,prev_action=action_sequence[:,j,i],sample=False)
                feat= dynamics_model.dynamics.get_feat(next_latents)
                predicted_rollout = dynamics_model.heads.decoder(feat)
                predicted_rollouts[:,:,j*rollout_length+i] = predicted_rollout['obs']._mode.reshape(num_steps,num_envs,obs_dim)
                std = next_latents['std'].mean(dim=-1)
                rollout_std += std.reshape(num_steps,num_envs)

        

        #predicted_rollouts: (num_envs,num_steps,repeat*rollout_length,obs_dim)->(num_envs,num_steps,repeat*rollout_length*obs_dim)
        predicted_rollouts = predicted_rollouts.reshape(num_steps,num_envs,repeat*rollout_length*obs_dim)

    return predicted_rollouts,rollout_std

def randomly_sample(actions, batch_length,repeat=5):
    num_steps,num_envs, action_dim = actions.size()
    action_sequence=torch.zeros((repeat,batch_length,action_dim)).to(actions.device)
    for i in range(repeat):
        env_idx = np.random.randint(0,num_envs)
        start_idx = np.random.randint(0,num_steps-batch_length)
        action_sequence[i] = actions[start_idx:start_idx+batch_length,env_idx]
    
    action_sequence = action_sequence.unsqueeze(0).repeat(num_steps*num_envs,1,1,1)
    return action_sequence
    
     
    
def compare_entropies(latent_ent,true_ent):
    #latent_ent: (num_steps,num_envs)
    #true_ent: (num_steps,num_envs)
    with torch.no_grad():
        normed_latent_ent = latent_ent/latent_ent.sum()
        normed_true_ent = true_ent/true_ent.sum()
        diff = normed_latent_ent-normed_true_ent
        over_estimation_indicies = torch.where(diff>0)
        over_estimation_frac =latent_ent[over_estimation_indicies].sum()/latent_ent.sum()
    return diff,over_estimation_frac

def precentile_basline(rewards,lower_percentile=0.05,upper_percentile=0.95):
    """
    Compute the percentile baseline for the PPO algorithm
    """
    rewards = np.array(rewards)
    lower_bound = np.percentile(rewards,lower_percentile*100)
    upper_bound = np.percentile(rewards,upper_percentile*100)
    
    in_range = np.logical_and(rewards >= lower_bound, rewards <= upper_bound)
    rewards = rewards[in_range]
    return torch.tensor(np.mean(rewards)).detach()

def update_rewards_hist(rewards,new_reward):
    """
    keeps a moving window of the last 100 episodes rewards
    """
    if len(rewards) > 20:
        #pop the first element
        rewards.pop(0)
    rewards.append(new_reward)
    return rewards

class MovingAvgRobust:
    def __init__(self, k: int, tensor_shape, device='cpu'):
        """
        Args:
            k (int): The maximum number of tensors to store (window size).
            tensor_shape (tuple): The shape of each tensor being stored.
            device (str): 'cpu' or 'cuda' to determine where the tensors are kept.
        """
        self.k = k
        # Preallocate a buffer; here we assume each new sample is of shape "tensor_shape".
        self.buffer = torch.empty((k, *tensor_shape), device=device)
        self.index = 0
        self.full = False

    def add(self, tensor: torch.Tensor):
        """Adds a new tensor to the buffer."""
        # Ensure the incoming tensor is on the same device and has the right shape.
        assert tensor.shape == self.buffer.shape[1:], "Tensor shape mismatch"
        self.buffer[self.index] = tensor.to(self.buffer.device)
        self.index = (self.index + 1) % self.k
        if self.index == 0:
            self.full = True

    def get_robust_avg(self, lower_quantile=0.05, upper_quantile=0.95):
        """Computes the average over the values falling within [5th, 95th] percentile range.

        Returns:
            torch.Tensor: The robust average.
        """
        # Select only the valid data (if the buffer isn’t yet full, use only the first "index" entries).
        if self.full:
            data = self.buffer
        else:
            data = self.buffer[:self.index]
        
        # Compute the per-element quantile thresholds along the first dimension (across samples)
        lower_bound = torch.quantile(data, lower_quantile, dim=0)
        upper_bound = torch.quantile(data, upper_quantile, dim=0)
        
        # Create a mask that is True for values within the threshold range
        mask = (data >= lower_bound) & (data <= upper_bound)
        
        # Sum only the in-range values and count them (avoid division by zero)
        masked_sum = torch.sum(data * mask, dim=0)
        count = torch.sum(mask, dim=0).float().clamp(min=1)
        
        # Compute the mean for each element, ignoring out-of-bound (masked) elements
        robust_avg = masked_sum / count
        
        return robust_avg
    

    
def discount(beta_decay,shape):
    num_steps, num_envs = shape
    discount = torch.zeros(shape)
    for i in range(num_steps):
        discount[i] = 1 *  beta_decay** i

    return discount

