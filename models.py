
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

# Hyperparams
# GAMMA = 0.95
# EPSILON = 1.0
# EPSILON_MIN = 0.01 
# EPSILON_DECAY = 0.995
# LEARNING_RATE = 0.001
# BATCH_SIZE = 64
# MEMORY_SIZE = 10000
# TARGET_UPDATE = 10
# HIDDEN_SIZE = 128 
# DROPOUT = 0.2
# NUM_GOODS = 12
# EPISODES = 100

class NeuralNetwork(nn.Module):
    def __init__(self, state_size, num_goods, action_size, hidden_size=128, dropout=0.1):
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
        q_values = torch.stack([head(encoding) for head in self.q_heads], dim=1)
        return q_values

class Memory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNetwork:

    def __init__(
        self,
        state_size,
        action_size,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay = 0.995,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        target_update=1000,
        memory_size=10000,
        episodes=1000,
        device='cpu'
    ):

        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        self.memory_size = memory_size
        self.episodes = episodes
        self.device = device

        # Separate policy network and target network so policy network is not trained on itself
        self.policy_net = NeuralNetwork(state_size, action_size).to(device)
        self.target_net = NeuralNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.memory = Memory(capacity=self.memory_size)
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        '''
        Epsilon-randomly choose random action or best action based on Q values.
        '''
        # Explore random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        # Exploit optimal action
        else:
            # with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()    

    def train(self):
        if self.memory.len < self.batch_size:
            return

        steps = 0

        for _ in range(self.episodes):

            # Optimize
            batch = self.memory.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.LongTensor(action_batch).unsqueeze(1)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(next_state_batch)
            done_batch = torch.FloatTensor(done_batch)

            # Calculate q value
            q_values_all = self.policy_net(state_batch) # (batch, num_goods, actions)
            action_batch = action_batch.unsqueeze(-1)   # (batch, num_goods, 1)
            q_values = q_values_all.gather(2, action_batch).squeeze(-1) # (batch, num_goods)

            with torch.no_grad():
                # max_next_q_values = self.target_net(next_state_batch).max(1)[0]
                # target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch) # TODO: Why this target in bellman?

                max_next_q_values = self.target_net(next_state_batch).max(2)[0]
                target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

            
            loss = self.loss(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update target net
            if steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            steps += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay + self.epsilon)
        
        # rewards.append(episode_reward)
    
    def store_transition(self, state, action, reward, next_state, done):
        
        self.memory.push(state, action, reward, next_state, done)
        return

    def save(self, filepath):
        torch.save({
            'q_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            return True
        return False
