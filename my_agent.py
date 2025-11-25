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

# Hyperparams
GAMMA = 0.95
EPSILON = 1.0
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
HIDDEN_SIZE = 128 
DROPOUT = 0.2
NUM_GOODS = 12

NAME = 'keyreg' # TODO: Please give your agent a NAME

class MyAgent(MyLSVMAgent):
    def __init__(self, name):
        super().__init__(name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = NUM_GOODS * 4 + 2 # state: [vals, min_bids, cur_prices, eligib, is_national, round_num] * num goods
        self.num_bid_levels = 5 
        self.action_size = self.num_bid_levels ** 3
        self.agent = DQNAgent(self.state_size, self.action_size, device=self.device)
        self.training = True
        
        model_path = path_from_local_root("models/dqn_agent.pth")
        if self.agent.load(model_path):
            print(f"Loaded pre-trained model from {model_path}")

    def setup(self):
        #TODO: Fill out with anything you want to initialize each auction
        self.prev_state = None
        self.prev_action = None
        self.round_utils = []
        self.round_num = 0
        return
        
    def get_state(self):

        # Get list of states with values, min bids, and prices
        valuations = self.get_valuations()
        min_bids = self.get_min_bids()
        cur_prices = min_bids.copy() # cur_prices = self.get_current_prices()
        eligibility = {good: 1 for good in valuations.keys()} # eligibility = self.get_eligibility()
        sorted_goods = sorted(valuations.keys())

        # Create state list
        state = []
        max_val = max(valuations.values()) if valuations else 1
        for good in sorted_goods[:NUM_GOODS]:
            state.append(valuations.get(good, 0) / max_val)
            state.append(min_bids.get(good, 0) / max_val)
            state.append(cur_prices.get(good, 0) / max_val) 
            state.append(eligibility.get(good, 1))

        state.append(1 if self.is_national_bidder() else 0)
        state.append(self.round_num / 100.0)

        return np.array(state, dtype=np.float32)
    
    def action_to_bids(self, action):
        valuations = self.get_valuations()
        min_bids = self.get_min_bids()
        bids = {}
        sorted_goods = sorted(valuations.items(), key=lambda x: x[1], reverse=True)
        
        # Action is int in range(num_bid_levels ** 3)
        # Each action represents a combination of bid multipliers for each of top 3 items
        # Bid multipliers on min_bid
        bid_multipliers = [1.0, 1.05, 1.1, 1.2, 1.5]
        num_bid_goods = 3 if self.is_national_bidder() else 2
        
        for i in range(min(num_bid_goods, len(sorted_goods))):
            good, val = sorted_goods[i]
            bid_level = (action // (self.num_bid_levels ** i)) % self.num_bid_levels
            if bid_level > 0:
                multiplier = bid_multipliers[bid_level]
                bid_amount = min_bids[good] * multiplier
                if val > bid_amount:
                    bids[good] = bid_amount
        
        return bids
    
    def national_bidder_strategy(self):
        # Get state, choose optimal action, get bid
        state = self.get_state()
        action = self.agent.select_action(state, training=self.training)
        self.prev_state, self.prev_action = state, action
        return self.action_to_bids(action)
    
    def regional_bidder_strategy(self):
        # Get state, choose optimal action, get bid
        state = self.get_state()
        action = self.agent.select_action(state, training=self.training)
        self.prev_state, self.prev_action = state, action
        
        # Filter for proximity goods
        all_bids = self.action_to_bids(action)
        proximity = self.get_goods_in_proximity()
        bids = {good: bid for good, bid in all_bids.items() if good in proximity}
        return bids
    
    def get_bids(self):
        if self.is_national_bidder():
            return self.national_bidder_strategy()
        else:
            return self.regional_bidder_strategy()
    
    def calculate_reward(self):
        cur_util = self.calc_total_utility()
        # reward = current util - prev util
        if len(self.round_utils) > 0:
            reward = cur_util - self.round_utils[-1]
        else:
            reward = cur_util
        
        self.round_utils.append(cur_util)
        return reward
    
    def update(self):
        self.round_num += 1
        if self.prev_state is not None and self.training:
            reward = self.calculate_reward()
            next_state = self.get_state()
            done = False
            self.agent.store_transition(
                self.prev_state,
                self.prev_action,
                reward,
                next_state,
                done
            )
            loss = self.agent.train()
            
            # if loss is not None and self.round_number % 10 == 0:
            #     print(f"Round {self.round_number}, Loss: {loss:.4f}, Epsilon: {self.dqn_agent.epsilon:.3f}")

    def teardown(self):
        #TODO: Fill out with anything you want to run at the end of each auction
        pass 

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=HIDDEN_SIZE, dropout=DROPOUT):
        super().__init__()

        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class Memory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_len(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=EPSILON, gamma=GAMMA, device='cpu'):
        # Training params
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

        # Separate q network and target network so q network is not trained on itself
        self.q_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.memory = Memory(MEMORY_SIZE)
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        # Randomly choose random action or best action based on q values
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def train(self):
        if self.memory.get_len() < BATCH_SIZE:
            return None
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        # Get batches
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)
        
        # Q learning update
        cur_q = self.q_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        loss = self.loss_fn(cur_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']
            return True
        return False

################### SUBMISSION #####################
my_agent_submission = MyAgent(NAME)
####################################################


def process_saved_game(filepath): 
    """ 
    Here is some example code to load in a saved game in the format of a json.gz and to work with it
    """
    print(f"Processing: {filepath}")
    
    # NOTE: Data is a dictionary mapping 
    with gzip.open(filepath, 'rt', encoding='UTF-8') as f:
        game_data = json.load(f)
        for agent, agent_data in game_data.items(): 
            if agent_data['valuations'] is not None: 
                # agent is the name of the agent whose data is being processed 
                agent = agent 
                
                # bid_history is the bidding history of the agent as a list of maps from good to bid
                bid_history = agent_data['bid_history']
                
                # price_history is the price history of the agent as a list of maps from good to price
                price_history = agent_data['price_history']
                
                # util_history is the history of the agent's previous utilities 
                util_history = agent_data['util_history']
                
                # util_history is the history of the previous tentative winners of all goods as a list of maps from good to winner
                winner_history = agent_data['winner_history']
                
                # elo is the agent's elo as a string
                elo = agent_data['elo']
                
                # is_national_bidder is a boolean indicating whether or not the agent is a national bidder in this game 
                is_national_bidder = agent_data['is_national_bidder']
                
                # valuations is the valuations the agent recieved for each good as a map from good to valuation
                valuations = agent_data['valuations']
                
                # regional_good is the regional good assigned to the agent 
                # This is None in the case that the bidder is a national bidder 
                regional_good = agent_data['regional_good']
            
            # TODO: If you are planning on learning from previously saved games enter your code below. 
            
            
        
def process_saved_dir(dirpath): 
    """ 
     Here is some example code to load in all saved game in the format of a json.gz in a directory and to work with it
    """
    for filename in os.listdir(dirpath):
        if filename.endswith('.json.gz'):
            filepath = os.path.join(dirpath, filename)
            process_saved_game(filepath)
            

if __name__ == "__main__":
    
    # Heres an example of how to process a singular file 
    # process_saved_game(path_from_local_root("saved_games/2024-04-08_17-36-34.json.gz"))
    # or every file in a directory 
    # process_saved_dir(path_from_local_root("saved_games"))

    print('########## NEW RUN ##############')
    
    ### DO NOT TOUCH THIS #####
    agent = MyAgent(NAME)
    arena = LSVMArena(
        num_cycles_per_player = 3,
        timeout=1,
        local_save_path="saved_games",
        players=[
            agent,
            MyAgent("CP - MyAgent"),
            MyAgent("CP2 - MyAgent"),
            MyAgent("CP3 - MyAgent"),
            MinBidAgent("Min Bidder"), 
            JumpBidder("Jump Bidder"), 
            TruthfulBidder("Truthful Bidder"), 
        ]
    )
    
    start = time.time()
    arena.run()
    end = time.time()
    print(f"{end - start} Seconds Elapsed")
