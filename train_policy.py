# train_policy

import functools
import os
import d4rl
import gym
import numpy as np
import torch
import tqdm
import wandb
from dataset import D4RL_dataset
from SRPO_model import SRPO
from utils import get_args, marginal_prob_std, parallel_simple_eval_policy

def train_policy(args, score_model, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_interval = 2
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):  
            data = data_loader.sample(args.policy_batchsize)
            loss = score_model.update_SRPO_policy(data)  
            avg_loss += loss  # Corrected to accumulate the returned loss
            num_items += 1
        avg_loss /= num_items  # Calculate mean loss outside the loop for efficiency
        tqdm_epoch.set_description(f'Average Loss: {avg_loss:.5f}')
        
        if (epoch % evaluation_interval == 0) or (epoch == n_epochs - 1):  # Evaluating on the first and last epoch, and periodically
            mean, std = parallel_simple_eval_policy(score_model.SRPO_policy.select_actions, args.env, seed = 0)
            args.run.log({"eval/rew_deter": mean, "info/policy_q": score_model.SRPO_policy.q.detach().cpu().numpy(), 
                          "info/lr": score_model.SRPO_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)

