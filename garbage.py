DQN Training Implementation for Spectrum Auctions





        # print('=====testing map_to_list========')
        # map = self.get_valuations()
        # map_as_list = self.map_to_list(map)
        # print('valuation list: ', map_as_list)
        # for i, good in enumerate('ABCDEFGHIJKLMNOPQR'):
        #     assert map_as_list[i] == map[good], 'Error: map_to_list is bugged.'
        # print('====Successful=====')




## First train
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
    cur_q = self.policy_net(state_batch).gather(1, action_batch)
    with torch.no_grad():
        next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
    
    loss = self.loss(cur_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
    self.optimizer.step()
    
    return loss.item()



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


def update(self):

    if self.prev_state is None or not self.training:
        return

    self.

    self.round_num += 1
    if self.prev_state is not None and self.training:
        reward = self.calculate_reward()
        next_state = self.get_state()
        done = False
        self.network.store_transition(
            self.prev_state,
            self.prev_action,
            reward,
            next_state,
            done
        )
        loss = self.agent.train()
        
        # if loss is not None and self.round_number % 10 == 0:
        #     print(f"Round {self.round_number}, Loss: {loss:.4f}, Epsilon: {self.dqn_agent.epsilon:.3f}")


class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, dropout=0.1):
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

            ## Choose action
            ## This might be done by TA agent TODO: Confirm
            # state = self.env.reset() # TODO: Get initial state
            # episode_reward = 0
            # action = self.select_action(state)
            # next_state, reward, done, _ = self.env.step(action)

            # # Store transition in memory
            # self.memory.push((state, action, reward, next_state, done))

            # # Update state
            # state = next_state
            # episode_reward += reward

            # Optimize
            batch = self.memory.sample(self.batch_size)
            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

            state_batch = torch.FloatTensor(state_batch)
            action_batch = torch.LongTensor(action_batch).unsqueeze(1)
            reward_batch = torch.FloatTensor(reward_batch)
            next_state_batch = torch.FloatTensor(next_state_batch)
            done_batch = torch.FloatTensor(done_batch)

            # Calculate q value
            q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

            with torch.no_grad():
                max_next_q_values = self.target_net(next_state_batch).max(1)[0]
                target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch) # TODO: Why this target in bellman?
            
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



def get_reward(self):
    cur_util = self.calc_total_utility()
    # reward = current util - prev util
    if len(self.round_utils) > 0:
        reward = cur_util - self.round_utils[-1]
    else:
        reward = cur_util
    
    self.round_utils.append(cur_util)
    return reward



    def save(self, filepath):
        torch.save({
            'q_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['q_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']
            return True
        return False


                # print('=====testing map_to_list========')
        # map = self.get_valuations()
        # map_as_list = self.map_to_list(map)
        # print('valuation list: ', map_as_list)
        # for i, good in enumerate('ABCDEFGHIJKLMNOPQR'):
        #     assert map_as_list[i] == map[good], 'Error: map_to_list is bugged.'
        # print('====Successful=====')

        # print('=====proximity_to_mask========')
        # prox_mask = self.proximity_to_mask(self.get_goods_in_proximity())
        # print(prox_mask)
        # assert prox_mask == [1.0] * 18, 'Error: prox_to_mask bugged'

        # prox_mask = self.proximity_to_mask(['A', 'B', 'C', 'E', 'Q', 'R'])
        # print(prox_mask)
        # assert prox_mask == [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 'Error: prox_to_mask 2'
        # print('====Successful=====')




        # print('prices: ', prices) #, len(prices))
        # print('min_bids: ', min_bids)
        # print('alloc: ', allocation) #, len(allocation))
        # print('is_nat: ', is_national) #, len(is_national))
        # print('round: ', round) #, len(round))
        # print('prox: ', proximity) #, len(proximity))
        # print('util: ', utility) #, len(utility))

        # print('num_goods', self.get_num_goods())
        # print('goods', self.get_goods())
        # print('tent alloc', self.get_tentative_allocation())

        # print('vals_as_arr: ', valuations, len(valuations)) #, len(valuations))
        # # print('get_vals: ', self.get_valuations()) #, len(valuations))
        # print('===get val for each good=====')
        # for good in self.get_goods():
        #     print(good, '\t', self.get_valuation(good))


                # print('prices: ', prices) #, len(prices))
        # print('min_bids: ', min_bids)
        # print('alloc: ', allocation) #, len(allocation))
        # print('is_nat: ', is_national) #, len(is_national))
        # print('round: ', round) #, len(round))
        # print('prox: ', proximity) #, len(proximity))
        # print('util: ', utility) #, len(utility))

        # # print('num_goods', self.get_num_goods())
        # print('goods', self.goods)
        # # print('tent alloc', self.get_tentative_allocation())




    def action_to_bids(self, actions: torch.Tensor) -> dict:
        '''
        Parameters
        ----------
        action: torch.Tensor
            Tensor of shape (num_goods,) with values 0-action_size
            for bid level action for each good.
                0: No bid
                1: Bid valuation
                2-6 : min_bid * multiplier in [1.0, 1.05, 1.1, 1.2, 1.5]

        Returns
        -------
        dict: Dictionary of good names to bids.
        '''
        min_bids = self.get_min_bids()
        valuations = self.get_valuations() 
        bids = {} 
        print('########action_to_bids ### actions: ', actions)
        for i, good in enumerate(self.goods):
            action = int(actions[i])
            if action == 0:
                continue
            if action == 1:
                bid = valuations[good]
            elif 2 <= action and action <= len(self.multipliers):
                bid = min_bids[good] * self.multipliers[action]
            else:
                raise Exception('Invalid action')
            if bid >= min_bids[good]:
                bids[good] = bid
        return bids


    def select_action(self, state, training=True):
        '''
        Epsilon-randomly choose random action or best action based on Q values.

        Args:
            state: torch.Tensor
                Tensor of concatenation of current state information
                (prices, valuations, allocations, etc)
        Returns
            int:
                Chosen action
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



            