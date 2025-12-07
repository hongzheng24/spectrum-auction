import unittest

from models import *


class IOTest(unittest.TestCase):
    
    def test_nn_shapes(self):
        net = NeuralNetwork(state_size=93, num_goods=18, action_size=5)
        
        # Single sample
        x1 = torch.randn(93)
        assert net(x1.unsqueeze(0)).shape == (1, 18, 5)
        
        # Batch
        x2 = torch.randn(32, 93)
        assert net(x2).shape == (32, 18, 5)
    
        # Empty batch
        x3 = torch.randn(0, 93)
        assert net(x3).shape == (0, 18, 5)

    def test_dqn_numerical_stability(self):
        pass
    
    def test_dqn_gradient(self):
        pass

    def test_dqn_train(self):
        dqnetwork = DQNetwork(state_size=93, num_goods=18, action_size=5)
        states = torch.randn(128, 93)
        actions = torch.randint(0, 5, (128, 18))
        rewards = torch.randn(128)
        next_states = torch.randn(128, 93)
        dones = torch.zeros(128)

        for i in range(128):
            dqnetwork.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # print(dqnetwork.memory.memory[0])

        loss = dqnetwork.train()
        print('loss: ', loss)
        pass

if __name__ == "__main__":
    unittest.main()
