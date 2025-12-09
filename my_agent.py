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

from models import DQNetwork

# Hyperparams
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01 
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 1000
HIDDEN_SIZE = 128 
DROPOUT = 0.0
NUM_GOODS = 18
EPISODES = 1000


SAVE_FREQ = 100
CHECKPOINT_DIR = 'checkpoints'
FILENAME = 'dqn_model_action_to_bid_logic_fix.pt'
FILEPATH = 'checkpoints/dqn_model_action_to_bid_logic_fix.pt'



############ TODO ########
# 1. Define state space
# 2. Define action space
# 3. Define reward
# 4. Implement some nuance
# 5. Writeup
MULTIPLIERS = {
    2: 1.0,
    3: 1.05,
    4: 1.1,
    5: 1.2,
    6: 1.5
}
STATE_SIZE = 18 * 5 + 1 * 3
ACTION_SIZE = len(MULTIPLIERS) + 2 # Bid nothing, bid valuation, or bid multiplier times min bid

################

NAME = 'keyreg' # TODO: Please give your agent a NAME

class MyAgent(MyLSVMAgent):
    def __init__(self, name):
        super().__init__(name)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE

        self.network = DQNetwork(
            STATE_SIZE,
            NUM_GOODS,
            ACTION_SIZE,
            hidden_size=HIDDEN_SIZE,
            epsilon=EPSILON,
            epsilon_min=EPSILON_MIN,
            epsilon_decay=EPSILON_DECAY,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            lr=LEARNING_RATE,
            target_update=TARGET_UPDATE,
            memory_size=MEMORY_SIZE,
            episodes=EPISODES,
            dropout=DROPOUT,
            device=self.device,
            checkpoint_dir=CHECKPOINT_DIR,
            filename=FILENAME,
            filepath=FILEPATH
        )
        # Try to load pre-trained model
        model_path = path_from_local_root(FILEPATH)
        if os.path.exists(model_path):
            self.network.load_checkpoint(model_path)
            self.training = False  # Use trained model without exploration
        else:
            self.training = True  # Train from scratch



        self.training = True
        self.goods = sorted(list(self.get_goods()))
        # print('SELF.GOODS: ', self.goods)

        self.multipliers = MULTIPLIERS
        self.action_size = ACTION_SIZE

        self.prev_state = None
        self.prev_action = None
        self.round_utils = []
        
        # TODO: Model save/load
        # model_path = path_from_local_root("models/dqn_agent.pth")
        # if self.network.load_checkpoint(model_path):
        #     print(f"Loaded pre-trained model from {model_path}")

    def setup(self):
        #TODO: Fill out with anything you want to initialize each auction
        self.prev_state = None
        self.prev_action = None
        self.round_utils = []
        return
    
    def national_bidder_strategy(self):
        # Get state, choose optimal action, get bid
        state = self.get_state().unsqueeze(0)
        action = self.network.select_action(state, training=self.training)
        self.prev_state, self.prev_action = state, action
        return self.action_to_bids(action)
    
    def regional_bidder_strategy(self):
        # Get state, choose optimal action, get bid
        state = self.get_state().unsqueeze(0)
        action = self.network.select_action(state, training=self.training)
        self.prev_state, self.prev_action = state, action
        
        # Filter for proximity goods
        all_bids = self.action_to_bids(action)
        proximity = self.get_goods_in_proximity()
        bids = {good: bid for good, bid in all_bids.items() if good in proximity}
        return bids
    
    def get_bids(self):
        if self.is_national_bidder():
            bids =  self.national_bidder_strategy()
        else:
            bids = self.regional_bidder_strategy()
        # print(bids)
        # bids = self.clip_bids(bids)
        # print(bids)

        # #####
        # if self.is_valid_bid_bund
        # #####

        if self.is_valid_bid_bundle(bids):
            return self.clip_bids(bids)
        assert self.is_valid_bid_bundle(bids) is True, 'Exception: Invalid bid!'
        return bids
    
    def update(self):
        if self.prev_state is None or not self.training:
            return

        reward = self.get_reward()
        next_state = self.get_state()
        done = False

        self.network.store_transition(
            self.prev_state,
            self.prev_action,
            reward,
            next_state,
            done
        )

        loss = self.network.train()

        return

    def teardown(self):
        #TODO: Fill out with anything you want to run at the end of each auction
        pass 

    def get_state(self) -> torch.Tensor():
        '''
        Returns:
            state: torch.Tensor()
                Tensor of shape (, state_size), containing ...
        '''

        # 1. prices = agent.get_current_prices()
        # 2. valuations = agent.get_valuations_as_array()
        # 3. allocation = binary mask of agent.get_tentative_allocation()
        # 4. margin = (valuations - prices) / scale
        # 5. is_national = 1.0 if agent.is_national_bidder() else 0.0
        # 6. round_progress = agent.get_current_round() / 500
        # 7. proximity = compute_proximity_mask(agent)
        # 8. utility = agent.calc_total_utility() / scale
        
        # 9. state = concatenate([
        #        prices, valuations, allocation, margin,
        #        is_national, round_progress, proximity, utility
        #    ])
        
        # 10. return torch.tensor(state, dtype=float32)
        scale = 100.0

        prices = self.get_current_prices().flatten() / 50.0 if self.get_current_prices() is not None else [0.0] * 18      # self.map_to_ndarray(self.current_prices_map()) #                    # 18
        valuations = self.map_to_ndarray(self.get_valuations()).flatten() / 30.0        # 18
        min_bids = self.map_to_ndarray(self.get_min_bids()).flatten() / 50.0            # 18
        allocation = self.set_to_list(self.get_tentative_allocation()) # TODO one hot encoding # 18
        # margin = (valuations - prices) / scale,
        is_national = [1.0 if self.is_national_bidder() else 0.0]      # 1
        round = [self.get_current_round() / 500.0]                              # 1
        proximity = self.proximity_to_mask(self.get_goods_in_proximity())                      # 18
        utility = [self.calc_total_utility() / 100.0]                             # 1

        # prev_util = [self.get_previous_util()]
        # prev_bid = self.get_previous_bid()
        # prev_winners = self.get_previous_winners()

        # print('prices: ', prices) #, len(prices))
        # print('valuations: ', valuations.shape)
        # print('min_bids: ', min_bids.shape)
        # print('alloc: ', allocation) #, len(allocation))
        # print('is_nat: ', is_national) #, len(is_national))
        # print('round: ', round) #, len(round))
        # print('prox: ', proximity) #, len(proximity))
        # print('util: ', utility) #, len(utility))

        state = np.concatenate((
            prices,
            valuations,
            min_bids,
            allocation,
            is_national,
            round,
            proximity,
            utility,
            # margin
        ))
        state= torch.tensor(state, dtype=torch.float32)
        return state

    def action_to_bids(self, actions: torch.Tensor) -> dict:
        '''
        Args:
            action: torch.Tensor
                Tensor of shape (num_goods,) with values 0-action_size
                for bid level action for each good.
                    0: No bid
                    1: Bid valuation
                    2-6 : min_bid * multiplier in [1.0, 1.05, 1.1, 1.2, 1.5]

        Returns:
            dict: Dictionary of good names to bids.
        '''
        min_bids = self.get_min_bids()
        valuations = self.get_valuations() 
        bids = {}
        for i, good in enumerate(self.goods):
            action = int(actions[i])
            if action == 0:
                # bids[good] = 0
                continue
            if action == 1:
                bid = valuations[good]
            elif 2 <= action and action < len(self.multipliers) + 2:
                bid = min_bids[good] * self.multipliers[action]
            else:
                raise Exception('Invalid action')
            
            bid = max(bid, min_bids[good])  # At least min_bid
            bid = min(bid, valuations[good])  # At most valuation
            
            if bid >= min_bids[good]:
                bids[good] = bid

            
            

        
        return bids

    def get_reward(self):
        return self.calc_total_utility()

    def map_to_list(self, map):
        '''
        Args:
            map: dict
                Mapping of goods (str A-R) to some utility value
                (tentative_allocation, valuations, etc)
        
        Returns:
            map_as_list: list
                List of utility value where i-th row corresponds
                to i-th good
        '''
        map_as_list = []
        for i, good in enumerate(self.goods):
            map_as_list.append(map[good])
        return map_as_list

    def set_to_list(self, init_set):
        set_as_list = [0.0] * 18
        for i, good in enumerate(self.goods):
            if good in init_set:
                set_as_list[i] = 1.0
        return set_as_list

    def proximity_to_mask(self, proximity_list):
        prox_mask = [0.0] * NUM_GOODS

        for good in proximity_list:
            good_idx = ord(good) - ord('A')
            prox_mask[good_idx] = 1.0
        return prox_mask

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