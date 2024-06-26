# train_policy

import functools
import os
import csv
import d4rl
import gym
import numpy as np
import torch
import tqdm
from dataset import D4RLDataset
from SRPO_model import SRPO
from utils import get_args, marginal_prob_std, parallel_simple_eval_policy
from logger import CustomLogger
import time

def train_policy(args, srpo_policy, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_interval = 2
    normalized_score = [['mean', 'std']]
    time_cost = np.zeros(n_epochs)
    for i, epoch in enumerate(tqdm_epoch):
        avg_loss = 0.
        num_items = 0
        start_time = time.time()
        for _ in range(10000):  
            data = data_loader.sample(args.policy_batchsize)
            loss = srpo_policy.update_SRPO_policy(data)  
            avg_loss += loss  # Corrected to accumulate the returned loss
            num_items += 1
        avg_loss /= num_items  # Calculate mean loss outside the loop for efficiency
        end_time = time.time()
        time_cost[i] = end_time - start_time
        
        tqdm_epoch.set_description(f'Average Loss: {avg_loss:.5f}')
        
        if (epoch % evaluation_interval == 0) or (epoch == n_epochs - 1):  # Evaluating on the first and last epoch, and periodically
            mean, std, time_eval, query_eval = parallel_simple_eval_policy(srpo_policy.SRPO_policy.select_actions, args.env, seed = 0)
            args.run.log(eval=mean, policy_q=srpo_policy.SRPO_policy.q.detach().cpu().numpy(), 
                          lr=srpo_policy.SRPO_policy_optimizer.state_dict()['param_groups'][0]['lr'])
            normalized_score.append([mean, std])
            if epoch == n_epochs - 1:
                time_eval_record = [['Query times', 'Time'], [query_eval, time_eval]]
                file_time = os.path.join("./SRPO_policy_models", str(args.expid), "eval_time.csv")
                with open(file_time, mode='w', newline='') as file_t:
                    writer = csv.writer(file_t)
                    writer.writerows(time_eval_record)
    
                

    # save policy
    torch.save(srpo_policy.state_dict(), os.path.join("./SRPO_policy_models", str(args.expid), "policy.pth"))
    
    # save normalized score
    filename = os.path.join("./SRPO_policy_models", str(args.expid), "normalized_score.csv")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(normalized_score)

    # save training time
    training_time = np.sum(time_cost)
    time_list = [['Gradient steps', 'Time'], [n_epochs*10000, training_time]]

    file_time = os.path.join("./SRPO_policy_models", str(args.expid), "training_time.csv")
    with open(file_time, mode='w', newline='') as file_t:
        writer = csv.writer(file_t)
        writer.writerows(time_list)
    



def training(args):
    if not os.path.exists("./SRPO_policy_models"):
        os.makedirs("./SRPO_policy_models")
    run_name = os.path.join("./SRPO_policy_models", str(args.expid))
    if not os.path.exists(run_name):
        os.makedirs(run_name)
    logger = CustomLogger(log_dir="./runs")
    for key, value in vars(args).items():
        logger.log(**{f'config/{key}': value})

    
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.run = logger
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device, beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn

    srpo_policy = SRPO(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    srpo_policy.q[0].to(args.device)

    if args.actor_load_path:
        try:
            ckpt = torch.load(args.actor_load_path, map_location=args.device)
            srpo_policy.load_state_dict({k: v for k, v in ckpt.items() if "diffusion_behavior" in k}, strict=False)
            print("Actor model loaded.")
        except FileNotFoundError:
            print("Actor model checkpoint not found.")

    if args.critic_load_path:
        try:
            ckpt = torch.load(args.critic_load_path, map_location=args.device)
            srpo_policy.q[0].load_state_dict(ckpt)
            print("Critic model loaded.")
        except FileNotFoundError:
            print("Critic model checkpoint not found.")

    dataset = D4RLDataset(args)

    print("Training SRPO policy...")
    train_policy(args, srpo_policy, dataset, start_epoch=0)
    print("Training completed.")
    logger.logger.close()
    

if __name__ == "__main__":
    args = get_args()
    training(args)


