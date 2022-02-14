import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import QNetwork
from memory import PrioritizedMemory, ReplayMemory

class Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 seed=0,
                 layer_units=[64, 64],
                 lr=6.25e-5,
                 gamma=0.997,
                 tau=1e-3,
                 multi_steps=3,
                 memory_size=1e5,
                 batch_size=64,
                 min_memory=64,
                 extensions=[]):

        # Saving variables
        self._state_size = state_size        #  environment state space
        self._action_size = action_size      #  environment discrete action space
        self._lr = lr                        #  learning rate (α)
        self._gamma = gamma                  #  discount factor (γ)
        self._tau = tau                      #  soft update factor (τ)
        self._batch_size = batch_size        #  size of every sample batch
        self._memory_size = memory_size      #  max number of experiences to store
        self._multi_steps = multi_steps      #  number of steps to activate learning step
        self._min_memory = min_memory        #  min number of experiences to start learning
        self._device = device                #  available processing device

        # Extensions
        self._prioritized_replay = 'Prioritized Replay' in extensions
        self._dueling = 'Dueling Network' in extensions
        self._double_dqn = 'Double DQN' in extensions

        # Initialize seeds
        self._random_seed = random.seed(seed)
        self._numpy_seed = np.random.seed(seed)

        # Q-Network
        self._target_qnetwork = QNetwork(state_size, action_size, layer_units=layer_units, seed=seed, dueling=self._dueling).to(device)
        self._local_qnetwork = QNetwork(state_size, action_size, layer_units=layer_units, seed=seed, dueling=self._dueling).to(device)
        self._optimizer = optim.Adam(self._local_qnetwork.parameters(), lr=lr)
        self._criterion = nn.MSELoss(reduce=(not self._prioritized_replay))

        # Replay memory
        MemoryStructure = PrioritizedMemory if self._prioritized_replay else ReplayMemory
        self._memory = MemoryStructure(int(memory_size), batch_size, device, seed=seed)

        # Multi_step counter
        self._t_step = 0

    def learn(self):
        if self._prioritized_replay:
            (states, actions, rewards, next_states, dones), weights, indices = self._memory.sample()
        else:
            states, actions, rewards, next_states, dones = self._memory.sample()

        # Get Q values for next states
        if self._double_dqn:
            Q_target_next = self._target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            Q_target_next = self._local_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q values for current states
        Q_targets = rewards + (self._gamma * Q_target_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self._local_qnetwork(states).gather(1, actions)

        # Compute loss
        loss = self._criterion(Q_expected, Q_targets)
        if self._prioritized_replay:
            loss = torch.mean(weights * loss)
            errors = torch.abs(Q_expected - Q_targets).cpu().data.numpy().squeeze(1)
        # Minimize the loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # Update target network
        if self._double_dqn: self.soft_update()
        if self._prioritized_replay: self._memory.update_priorities(indices, errors)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._local_qnetwork.eval()
        with torch.no_grad():
            Q_values = self._local_qnetwork(state)
        self._local_qnetwork.train()

        if random.random() > eps:
            action = np.argmax(Q_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self._action_size))

        return action

    def step(self, state, action, reward, next_state, done):
        # Save MDP tuple in memory
        self._memory.save(state, action, reward, next_state, done)

        # Multi step logic
        self._t_step = (self._t_step + 1) % self._multi_steps
        if self._t_step == 0 and len(self._memory) >= self._min_memory:
            self.learn()

    def soft_update(self):
        for target_param, local_param in zip(self._target_qnetwork.parameters(), self._local_qnetwork.parameters()):
            target_param.data.copy_(self._tau*local_param.data + (1.0-self._tau)*target_param.data)
