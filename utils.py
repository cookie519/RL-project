import argparse
import d4rl
import gym
import numpy as np
import torch
import time 
import pandas as pd
import os

temperature_coefficients = {"antmaze-medium-play-v2": 0.08, "antmaze-umaze-v2": 0.02, "antmaze-umaze-diverse-v2": 0.04, 
                            "antmaze-medium-diverse-v2": 0.05, "antmaze-large-diverse-v2": 0.05, "antmaze-large-play-v2": 0.06, 
                            "hopper-medium-expert-v2": 0.01, "hopper-medium-v2": 0.05, "hopper-medium-replay-v2": 0.2, 
                            "walker2d-medium-expert-v2": 0.1, "walker2d-medium-v2": 0.05, "walker2d-medium-replay-v2": 0.5, 
                            "halfcheetah-medium-expert-v2": 0.01, "halfcheetah-medium-v2": 0.2, "halfcheetah-medium-replay-v2": 0.2}

def marginal_prob_std(t, device = "cuda", beta_1 = 20.0, beta_0 = 0.1) -> tuple:
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$."""
    t_tensor = t.clone().detach()
    log_mean_coeff = -0.25 * t_tensor ** 2 * (beta_1 - beta_0) - 0.5 * t_tensor * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std

def parallel_simple_eval_policy(policy_fn, env_name, seed, eval_episodes = 20):
    """
    Evaluate a policy function in parallel across multiple episodes and environments.

    Args:
    - policy_fn: A function that takes state tensors and outputs actions.
    - env_name: Name of the Gym environment.
    - seed: Random seed for environment initialization.
    - eval_episodes: Number of parallel episodes to run.

    Returns:
    - Tuple containing the mean and standard deviation of the normalized scores.
    """
    environments = [gym.make(env_name) for _ in range(eval_episodes)]
    for i, env in enumerate(environments):
        env.seed(seed + 1001 + i)
        env.buffer_state = env.reset()
        env.buffer_return = 0.0

    time_cost = 0.0
    query_times = 0
    # Process environments in parallel
    active_envs = environments.copy()
    while active_envs:
        states = np.array([env.buffer_state for env in active_envs])
        states_tensor = torch.tensor(states, device="cuda", dtype=torch.float32)
        start_time = time.time()
        with torch.no_grad():
            actions = policy_fn(states_tensor).detach().cpu().numpy()
        end_time = time.time()
        time_cost += end_time - start_time
        query_times += 1
        
        next_envs = []
        for env, action in zip(active_envs, actions):
            state, reward, done, _ = env.step(action)
            env.buffer_return += reward
            env.buffer_state = state
            if not done:
                next_envs.append(env)
        active_envs = next_envs

    # Calculate normalized scores
    normalized_scores = [d4rl.get_normalized_score(env_name, env.buffer_return) for env in environments]
    mean_score = np.mean(normalized_scores)
    std_score = np.std(normalized_scores)

    return mean_score, std_score, time_cost, query_times

def get_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2")
    parser.add_argument("--seed", default=0, type=int)  
    parser.add_argument("--expid", default="default", type=str)    
    parser.add_argument("--device", default="cuda", type=str)      
    parser.add_argument("--save_model", default=1, type=int)       
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--beta', type=float, default=None)       
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--critic_load_path', type=str, default=None)
    parser.add_argument('--policy_batchsize', type=int, default=256)              
    parser.add_argument('--actor_blocks', type=int, default=3)     
    parser.add_argument('--z_noise', type=int, default=1)
    parser.add_argument('--WT', type=str, default="VDS")
    parser.add_argument('--q_layer', type=int, default=2)
    parser.add_argument('--n_policy_epochs', type=int, default=100)
    parser.add_argument('--policy_layer', type=int, default=None)
    parser.add_argument('--critic_load_epochs', type=int, default=150)
    parser.add_argument('--regq', type=int, default=0)
    print("**************************")
    args = parser.parse_known_args()[0]
    if args.debug:
        args.actor_epoch = 1
        args.critic_epoch = 1
    if args.policy_layer is None:
        args.policy_layer = 4 if "maze" in args.env else 2
    if "maze" in args.env:
        args.regq = 1
    if args.beta is None:
        args.beta = temperature_coefficients[args.env]
    print(args)
    return args


def plot_get_args() -> argparse.Namespace:
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2")
    print("**************************")
    args = parser.parse_known_args()[0]
    print(args)
    return args
    

def read_score_data(model_name='SRPO_policy_models', env="halfcheetah-medium-expert-v2", seeds=[0,1,2]):
  mean = []
  std = []
  for seed in seeds:
    expid = env + '-baseline-seed' + str(seed)
    filename = os.path.join(".", model_name, expid, "normalized_score.csv")
    df = pd.read_csv(filename)
    if 'mean' in df.columns and 'std' in df.columns:  # Validate column names
        mean.append(df['mean'].to_numpy())
        std.append(df['std'].to_numpy())
    

  mean = np.array(mean)
  std = np.array(std)
  
  overall_mean = np.mean(mean, axis=0)
  mean_variance = np.mean((mean - overall_mean)**2, axis=0)
  average_variances = np.mean(std**2, axis=0)
  overall_variance = average_variances + mean_variance
  overall_std_deviation = np.sqrt(overall_variance)

  return overall_mean, overall_std_deviation


def plot_training_performance(env="halfcheetah-medium-expert-v2", seeds=[0,1,2]):
  srpo_mean, spro_std = read_score_data('SRPO_policy_models', env=env, seeds=seeds)
  #diffql_mean, diffql_std = read_score_data('Diff_ql_models', env=env, seeds=seeds)

  x_srpo = np.arange(srpo_mean.shape[0]) * 20
  #x_diffql = np.arange(diffql_mean.shape[0]) * 20

  plt.figure(figsize=(10, 6))
  plt.plot(x_srpo, srpo_mean, label='SRPO', color='orangered')
  #plt.plot(x_diffql, diffql_mean, label= 'Diffusion-QL', color='c')
  #plt.plot(x_diffusion, y_diffusion, label='IQL', color='green')

  plt.fill_between(x_srpo, srpo_mean - spro_std, srpo_mean + spro_std, color='orangered', alpha=0.1)
  #plt.fill_between(x_diffql, diffql_mean - diffql_std, diffql_mean + diffql_std, color='c', alpha=0.1)
  #plt.fill_between(x_diffusion, y_diffusion - variance, y_diffusion + variance, color='green', alpha=0.1)

# Titles and labels
  plt.title(env)
  plt.xlabel('Policy Gradient Steps (k)')
  plt.ylabel('Normalized Score')

# Legend
  plt.legend()
  plt.savefig(f"{env}.png")
  plt.show()
  

  
  
    
  






  
