import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from test_script import QNetwork

# set seeds
SEEDS = [0, 1, 2]

# select device for training
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
    
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
            self.pos += 1
            if self.pos >= self.capacity:
                self.pos = 0

    def sample(self, batch_size, device=DEVICE):
        transitions = random.sample(self.buffer, batch_size)
        states      = [t[0] for t in transitions]
        actions     = [t[1] for t in transitions]
        rewards     = [t[2] for t in transitions]
        next_states = [t[3] for t in transitions]
        dones       = [t[4] for t in transitions]

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# Helper Functions
def set_seed(seed):
    """
    Set random seeds in: torch, numpy, and random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_epsilon(episode, decay_episodes, eps_start=1.0, eps_end=0.05):
    """
    Linearly decays epsilon,
    to promote exploration first and then exploitation
    """
    if episode >= decay_episodes:
        return eps_end
    return eps_start - (eps_start - eps_end) * (episode / decay_episodes)

def select_action(state, q_network, epsilon, action_dim, device=DEVICE):
    """
    Epsilon-greedy action selection.
    """
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(q_network(state_tensor).argmax())

def update_dqn(q_network, target_network, optimizer, replay_buffer,
               batch_size=64, gamma=0.99, loss_fn=None, device=DEVICE):
    """
    Performs one DQN gradient update step using experience replay and target network.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device=device)

    # Q(s, a) for chosen actions
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        max_next_q = target_network(next_states).max(1)[0]
        targets = rewards + (1.0 - dones) * gamma * max_next_q

    loss = loss_fn(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



# Main Training Loop
def train(get_pole_length_fn, seed, strategy_name, n_episodes=750,
          learning_rate=1e-3, gamma=0.99, batch_size=64,
          buffer_size=50000, min_buffer_size=1000, target_update_freq=500,
          eps_start=1.0, eps_end=0.05, eps_decay_episodes=None,
          weights_dir="weights", device=DEVICE):
    """
    Trains a DQN agent with the given pole length selection function and random seed.
    """
    os.makedirs(weights_dir, exist_ok=True)

    if eps_decay_episodes is None:
        eps_decay_episodes = n_episodes / 2

    set_seed(seed)

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    loaded_model = QNetwork(state_dim, action_dim).to(device) # used for training
    target_network = QNetwork(state_dim, action_dim).to(device) # for the loss calculation
    target_network.load_state_dict(loaded_model.state_dict())
    target_network.eval()

    optimizer = optim.Adam(loaded_model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_size)

    total_steps = 0

    print(f"\nStart training: {strategy_name}, Seed {seed}, Episodes {n_episodes}")

    env.reset(seed=seed)

    for episode in range(n_episodes):
        pole_length = float(get_pole_length_fn(episode, n_episodes))
        env.unwrapped.length = pole_length


        state = env.reset()
        state = state[0]
        epsilon = get_epsilon(episode, eps_start=eps_start, eps_end=eps_end, decay_episodes=eps_decay_episodes)

        while True:
            action = select_action(state, loaded_model, epsilon, action_dim, device=device)
            next_state, reward, terminated, truncated, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, float(terminated))

            state = next_state

            total_steps += 1

            # Update Q-network if enough transitions collected
            if len(replay_buffer) >= min_buffer_size:
                update_dqn(loaded_model, target_network, optimizer, replay_buffer,
                           batch_size=batch_size, gamma=gamma, device=device)

            # Update target network periodically
            if total_steps % target_update_freq == 0:
                target_network.load_state_dict(loaded_model.state_dict())

            if terminated or truncated:
                break


    # Save model weights
    weight_path = os.path.join(weights_dir, f"{strategy_name}_seed{seed}.pth")
    torch.save(loaded_model.to("cpu").state_dict(), weight_path) # save model on CPU because it was trained on MPS
    loaded_model.to(device)
    print(f"Saved model to: {weight_path}")

    env.close()
    return loaded_model

