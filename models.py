
from agt_server.agents.base_agents.lsvm_agent import MyLSVMAgent
from agt_server.local_games.lsvm_arena import LSVMArena
from agt_server.agents.test_agents.lsvm.min_bidder.my_agent import MinBidAgent
from agt_server.agents.test_agents.lsvm.jump_bidder.jump_bidder import JumpBidder
from agt_server.agents.test_agents.lsvm.truthful_bidder.my_agent import TruthfulBidder
import time
import os
import random
import gzip
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from path_utils import path_from_local_root

from typing import Tuple

CHECKPOINT_DIR = 'checkpoints'
FILENAME = 'dqn_model.pt'
FILEPATH = 'checkpoints/dqn_model.pt'


class NeuralNetwork(nn.Module):
    def __init__(self, state_size, num_goods, action_size, hidden_size=64, dropout=0.1):
        super().__init__()
        
        self.state_size = state_size
        self.num_goods = num_goods
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.encoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
            for _ in range(num_goods)
        ])
        
    def forward(self, state):
        '''
        Foward pass. Returns q values for each action for each good.
        Output shape: [batch_size, num_goods, action_size]
        State -> Encoder -> Heads * num_goods.
        State to encoding to num_goods heads.
        '''
        encoding = self.encoder(state)
        q_values = torch.stack([head(encoding) for head in self.q_heads], dim=1) #.squeeze(2)
        return q_values

class Memory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).squeeze(1),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

    @property
    def len(self):
        return len(self.memory)

class DQNetwork:

    def __init__(
        self,
        state_size,
        num_goods,
        action_size,
        hidden_size=64,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay = 0.995,
        batch_size=32,
        gamma=0.99,
        lr=0.001,
        target_update=1000,
        memory_size=10000,
        episodes=1000,
        dropout=0.0,
        save_freq=100,
        checkpoint_dir='',
        filename='',
        filepath='',
        device='cpu'
    ):

        # Hyperparameters
        self.state_size = state_size
        self.num_goods = num_goods
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        self.memory_size = memory_size
        self.episodes = episodes
        self.dropout = dropout
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.filepath = filepath
        self.device = device

        self.save_freq = save_freq

        # Separate policy network and target network so policy network is not trained on itself
        self.policy_net = NeuralNetwork(
            state_size=self.state_size,
            num_goods=self.num_goods,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(device)
        self.target_net = NeuralNetwork(
            state_size=self.state_size,
            num_goods=self.num_goods,
            action_size=self.action_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.memory = Memory(capacity=self.memory_size)
        self.steps = 0

        self.loss_history = []
        
    def select_action(self, state, training=True):
        '''
        Epsilon-randomly choose random action or best action based on Q values.

        Args:
            state: torch.Tensor
                Tensor of concatenation of current state information
                (prices, valuations, allocations, etc)
        Returns
            torch.Tensor
                Action tensor of shape (num_goods,)
        '''
        # Explore random action
        if training and random.random() < self.epsilon:
            action = torch.randint(
                0, self.action_size,
                (self.num_goods,),
                dtype=torch.long
            )
            return action
        # Exploit optimal action
        else:
            # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # q_values = self.policy_net(state)
            # return torch.argmax(q_values).item() 
            with torch.no_grad():
                q_values = self.policy_net(state)
                actions = q_values.argmax(dim=-1).squeeze(0).cpu()
                return actions

    def train(self):
        if self.memory.len < self.batch_size:
            return

        # Optimize
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device).long()
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Calculate q value
        q_values_all = self.policy_net(state_batch) # (batch, num_goods, actions)
        action_batch = action_batch.unsqueeze(-1)   # (batch, num_goods, 1)
        # q_values = q_values_all.gather(2, action_batch).squeeze(-1) # (batch, num_goods)
        q_values = torch.gather(q_values_all, 2, action_batch).squeeze(-1)

        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(2)[0]
            target_q_values = reward_batch.unsqueeze(1) + self.gamma * max_next_q_values * (1 - done_batch.unsqueeze(1))

        loss = self.loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target net
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.steps % self.save_freq == 0:
            self.save_checkpoint()

        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        self.steps += 1
        
        # rewards.append(episode_reward)
        self.loss_history.append(loss)
        return loss
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        return

    def save_checkpoint(self, dir: str=CHECKPOINT_DIR, filename=FILENAME):
        """Save training checkpoint."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'loss_history': self.loss_history,
            # 'reward_history': self.reward_history,
            'hyperparameters': {
                'state_dim': self.state_size,
                'num_goods': self.num_goods,
                'actions_per_good': self.action_size,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
            }
        }
        os.makedirs(dir, exist_ok=True)
        filepath = os.path.join(dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str=CHECKPOINT_DIR):
        """Load training checkpoint."""
        if not os.path.exists(filepath):
            print(f"Checkpoint not found: {filepath}")
            return None
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.loss_history = checkpoint.get('loss_history', [])
        # self.reward_history = checkpoint.get('reward_history', [])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from step {self.steps}")