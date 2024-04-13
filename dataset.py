import d4rl
import gym
import torch

def compute_episode_returns(dataset, max_episode_steps):
    """Compute the total returns for episodes, ensuring that the maximum steps per episode is not exceeded."""
    episode_returns = []
    total_length = 0
    current_return, current_length = 0.0, 0

    for reward, terminal in zip(dataset['rewards'], dataset['terminals']):
        current_return += float(reward)
        current_length += 1
        if terminal or current_length == max_episode_steps:
            episode_returns.append(current_return)
            total_length += current_length
            current_return, current_length = 0.0, 0

    assert total_length == len(dataset['rewards']), "Sum of episode lengths should match total number of rewards"
    return min(episode_returns), max(episode_returns)

class D4RLDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        env = gym.make(args.env)
        data = d4rl.qlearning_dataset(env)

        # Data pre-processing
        self.states = torch.tensor(data['observations'], dtype=torch.float32, device=self.device)
        self.actions = torch.tensor(data['actions'], dtype=torch.float32, device=self.device)
        self.next_states = torch.tensor(data['next_observations'], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(data['rewards'], dtype=torch.float32, device=self.device).view(-1, 1)
        self.terminals = torch.tensor(data['terminals'], dtype=torch.float32, device=self.device).view(-1, 1)
        
        self.rewards = self.scale_rewards(rewards, args.env, data)
        self.length = self.states.size(0)
        self.current_index = 0
        print("Data loaded")

    def scale_rewards(self, rewards, env_name, dataset):
        """Apply environment-specific scaling to rewards."""
        if "antmaze" in env_name:
            return rewards - 1.0
        elif "locomotion" in env_name:
            min_reward, max_reward = compute_episode_returns(dataset, 1000)
            normalized_rewards = rewards / (max_reward - min_reward)
            return normalized_rewards * 1000
        return rewards

    def __getitem__(self, index):
        index = index % self.length
        return { 's': self.states[index], 'a': self.actions[index], 'r': self.rewards[index],
                 's_': self.next_states[index], 'd': self.terminals[index] }

    def shuffle_data(self):
        indices = torch.randperm(self.length, device=self.device)
        self.states, self.actions, self.rewards, self.next_states, self.terminals = (
            x[indices] for x in (self.states, self.actions, self.rewards, self.next_states, self.terminals))

    def sample(self, batch_size):
        if self.current_index + batch_size > self.length:
            self.current_index = 0
        if self.current_index == 0:
            self.shuffle_data()
        batch_indices = slice(self.current_index, self.current_index + batch_size)
        batch_data = { 's': self.states[batch_indices], 'a': self.actions[batch_indices],
                       'r': self.rewards[batch_indices], 's_': self.next_states[batch_indices],
                       'd': self.terminals[batch_indices] }
        self.current_index += batch_size
        return batch_data

    def __len__(self):
        return self.length
