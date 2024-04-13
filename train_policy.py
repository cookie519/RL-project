# train_policy

import functools
import os
import csv
import d4rl
import gym
import numpy as np
import torch
import tqdm
import wandb
from dataset import D4RLDataset
from SRPO_model import SRPO
from utils import get_args, marginal_prob_std, parallel_simple_eval_policy

def train_policy(args, srpo_policy, data_loader, start_epoch=0):
    n_epochs = args.n_policy_epochs
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_interval = 2
    normalized_score = [['mean', 'std']]
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(10000):  
            data = data_loader.sample(args.policy_batchsize)
            loss = srpo_policy.update_SRPO_policy(data)  
            avg_loss += loss  # Corrected to accumulate the returned loss
            num_items += 1
        avg_loss /= num_items  # Calculate mean loss outside the loop for efficiency
        tqdm_epoch.set_description(f'Average Loss: {avg_loss:.5f}')
        
        if (epoch % evaluation_interval == 0) or (epoch == n_epochs - 1):  # Evaluating on the first and last epoch, and periodically
            mean, std = parallel_simple_eval_policy(srpo_policy.SRPO_policy.select_actions, args.env, seed = 0)
            args.run.log({"eval/rew_deter": mean, "info/policy_q": srpo_policy.SRPO_policy.q.detach().cpu().numpy(), 
                          "info/lr": srpo_policy.SRPO_policy_optimizer.state_dict()['param_groups'][0]['lr']}, step=epoch+1)
            normalized_score.append([mean, std])

    return normalized_score



def training(args):
    for dir in ["./SRPO_policy_models"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    if not os.path.exists(os.path.join("./SRPO_policy_models", str(args.expid))):
        os.makedirs(os.path.join("./SRPO_policy_models", str(args.expid)))
    run = wandb.init(project="SRPO_policy", name=str(args.expid))
    wandb.config.update(args)
    
    
    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.run = run
    
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
    normalized_score = train_policy(args, srpo_policy, dataset, start_epoch=0)
    print("Training completed.")
    run.finish()

    filename = './SRPO_data/Score/SRPO-' + args.env + 'seed' + str(args.seed)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(normalized_score)



if __name__ == "__main__":
    args = get_args()
    training(args)


