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

        loss = dqnetwork.train()
        # print('Loss 1: ', loss)
        # loss = dqnetwork.train()
        # print('Loss 2: ', loss)

        # for i in range(100):
        #     loss = dqnetwork.train()
        #     print(f'Loss {i}: ', loss)
        pass

    def test_dqn_loss_decreases(self):
        torch.manual_seed(42)
        random.seed(42)

        state_size = 93
        num_goods = 18
        action_size = 5

        dqn = DQNetwork(state_size=93, num_goods=18, action_size=5)

        # Add samples to DQNetwork memory
        for i in range(500):
            state = torch.randn(state_size)
            action = torch.randint(0, action_size, (num_goods,))
            reward = random.uniform(-10, 10)
            next_state = torch.randn(state_size)
            done = random.random() < 0.1
            dqn.store_transition(state, action, reward, next_state, done)


        losses = []
        for i in range(200):
            loss = dqn.train()
            # print(f'i: ', i, '\t \t Loss: ', loss)
            if loss is not None:
                losses.append(loss.item())

        assert len(losses) > 0, "No training occurred (memory too small?)"

        initial_loss = losses[0]
        final_loss = losses[-1]
        print(f"\n=== NeuralNetwork Overfit Test ===")
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Final loss:   {final_loss:.6f}")
        print(f"Reduction:    {(1 - final_loss/initial_loss)*100:.1f}%")

        # Should reduce loss by at least 90%
        assert final_loss < initial_loss * 0.1, \
            f"Failed to overfit. Loss: {initial_loss:.4f} → {final_loss:.4f}"
        
        # # Final loss should be very small
        # assert final_loss < 0.01, \
        #     f"Final loss too high: {final_loss:.4f}"


        window = 20
        first_avg = sum(losses[:window]) / window
        last_avg = sum(losses[-window:]) / window
        
        print(f"\n=== DQNetwork Training Results ===")
        print(f"Total training steps: {len(losses)}")
        print(f"First {window} avg loss: {first_avg:.6f}")
        print(f"Last {window} avg loss:  {last_avg:.6f}")
        print(f"Reduction: {(1 - last_avg/first_avg)*100:.1f}%")
        
        assert last_avg < first_avg, \
            f"Loss did not decrease: {first_avg:.4f} → {last_avg:.4f}"

    def test_overfit_fixed_transitions(self):
        """
        Critical test: Can DQNetwork memorize a small set of transitions?
        """
        torch.manual_seed(42)
        random.seed(42)
        
        state_size = 93
        num_goods = 18
        action_size = 5
        
        # Create DQN with settings optimized for overfitting
        dqn = DQNetwork(
            state_size=state_size,
            num_goods=num_goods,
            action_size=action_size,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            batch_size=8,          # Small batch = entire dataset
            gamma=0.99,
            lr=1e-3,               # Higher LR for faster convergence
            target_update=50,
            memory_size=100,
            device='cpu'
        )
        
        # Create exactly 8 fixed transitions (same as batch size)
        print("\n=== Creating Fixed Transitions ===")
        fixed_transitions = []
        
        for i in range(8):
            state = torch.randn(state_size)
            action = torch.randint(0, action_size, (num_goods,))
            reward = float(i)  # Unique rewards
            next_state = torch.randn(state_size)
            done = (i == 7)  # Last one is terminal
            
            fixed_transitions.append((state, action, reward, next_state, done))
            dqn.store_transition(state, action, reward, next_state, done)
        
        # Train on same 8 transitions repeatedly
        print("\n=== Overfitting Fixed Transitions ===")
        losses = []
        
        for i in range(500):
            loss = dqn.train()
            if loss is not None:
                losses.append(loss.item())
                dqn.steps += 1
        
        initial_loss = sum(losses[:10]) / 10
        final_loss = sum(losses[-10:]) / 10
        
        print(f"Initial loss (first 10): {initial_loss:.6f}")
        print(f"Final loss (last 10):    {final_loss:.6f}")
        print(f"Reduction: {(1 - final_loss/initial_loss)*100:.1f}%")
        
        # Should be able to significantly reduce loss
        assert final_loss < initial_loss * 0.5, \
            f"Should reduce loss by 50%+ when overfitting. Got {initial_loss:.4f} → {final_loss:.4f}"

    def test_train_parameter_update(self):
        net = NeuralNetwork(state_size=93, num_goods=18, action_size=5)
        target_net = NeuralNetwork(state_size=93, num_goods=18, action_size=5)
        target_net.load_state_dict(net.state_dict())
        
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        
        # Store initial params
        initial_params = {k: v.clone() for k, v in net.named_parameters()}
        
        # Training step
        states = torch.randn(32, 93)
        actions = torch.randint(0, 5, (32, 18))
        rewards = torch.randn(32)
        next_states = torch.randn(32, 93)
        dones = torch.zeros(32)
        
        # Compute loss (simplified)
        q_values = net(states)
        q_taken = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q = target_net(next_states).max(dim=-1).values
            target = rewards.unsqueeze(1) + 0.99 * next_q * (1 - dones.unsqueeze(1))
        
        loss = ((q_taken - target) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check params changed
        changed = any(
            not torch.equal(v, initial_params[k])
            for k, v in net.named_parameters()
        )
        assert changed, "Parameters did not update"

    def test_gradient_flow(self):
        net = NeuralNetwork(state_size=93, num_goods=18, action_size=5)
        
        x = torch.randn(32, 93, requires_grad=True)
        out = net(x)
        loss = out.sum()
        loss.backward()
        
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            
            grad_norm = param.grad.norm().item()
            assert grad_norm > 1e-10, f"Vanishing gradient for {name}"
            assert grad_norm < 1e6, f"Exploding gradient for {name}"
if __name__ == "__main__":
    unittest.main()
