from abc import ABC
import torch
import data_hyperparameters
import datetime
import matplotlib.pyplot as plt
from math import nan, log
import numpy as np
import os

if not os.path.exists('learning_curves/'):
    os.mkdir('learning_curves/')


def argmax(arr):  # I do this just because I want to randomly break ties, otherwise you could just use np.argmax()
    top = float('-inf')
    ties = []
    for i in range(len(arr)):
        if arr[i] > top:
            top = arr[i]
            ties = []
        if arr[i] == top:
            ties.append(i)
    return np.random.choice(ties)


class Replay:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.insertion_index = 0

    def __len__(self):
        return len(self.data)

    def adjust_beta(self):
        pass

    def update_priorities(self, batch_indices, batch_priorities):
        pass

    def add(self, state, action, reward, next_state, done):
        data_point = (state, action, reward, next_state, done)
        if len(self) < self.max_size:
            self.data.append(data_point)
        else:
            self.data[self.insertion_index] = data_point
        self.insertion_index = (self.insertion_index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.choice(len(self), min(batch_size, len(self)), replace=False)
        batch = list(zip(*[self.data[index] for index in indices]))
        return batch[0], batch[1], batch[2], batch[3], batch[4], None, None


class PrioritisedReplay(Replay):
    def __init__(self, max_size, beta=data_hyperparameters.INITIAL_BETA,
                 probability_power=data_hyperparameters.PROBABILITY_POWER):
        super().__init__(max_size)
        self.probability_power = probability_power
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.beta = beta

    def adjust_beta(self):
        self.beta += 1e-3
        if self.beta > 1.:
            self.beta = 1

    def update_priorities(self, batch_indices, batch_priorities):
        for index, priority in zip(batch_indices, batch_priorities):
            self.priorities[index] = priority

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if len(self) > 0 else 1.
        self.priorities[self.insertion_index] = max_priority
        super().add(state, action, reward, next_state, done)

    def sample(self, batch_size):
        if len(self) == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self)]
        probabilities = priorities ** self.probability_power
        probabilities /= np.sum(probabilities)
        indices = np.random.choice(len(self), min(batch_size, len(self)), replace=False, p=probabilities)
        batch = list(zip(*[self.data[index] for index in indices]))
        total = len(self)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= np.max(weights)
        weights = np.array(weights, dtype=np.float32)
        return batch[0], batch[1], batch[2], batch[3], batch[4], indices, weights


class DoubleDQN:
    def __init__(self, epsilon, discount_factor, state_dimension, num_actions,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE, use_cuda=data_hyperparameters.USE_CUDA):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.state_dimension = state_dimension
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.online_network = torch.nn.Sequential(torch.nn.Linear(state_dimension, hidden_size), torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_size, num_actions))
        self.target_network = torch.nn.Sequential(torch.nn.Linear(state_dimension, hidden_size), torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_size, num_actions))
        if use_cuda:
            self.online_network.cuda()
            self.target_network.cuda()
        self.synchronise()

    def synchronise(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def adjust_epsilon(self, epsilon_decay_factor=data_hyperparameters.EPSILON_DECAY_FACTOR):
        self.epsilon *= epsilon_decay_factor

    def get_value(self, states):
        with torch.no_grad():
            predicted_q_values = self.online_network(states)
            selected_actions = torch.argmax(predicted_q_values, dim=1)
            batch_size = states.shape[0]
            values = self.target_network(states)[range(batch_size), selected_actions]
        return values

    def calculate_temporal_difference_loss(self, states, actions, rewards, next_states, dones, weights):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            weights = torch.ones_like(rewards, dtype=torch.float32, device=self.device)
        batch_size = states.shape[0]
        predicted_q_values = self.online_network(states)[range(batch_size), actions]
        with torch.no_grad():
            temporal_difference_target = torch.where(dones, rewards,
                                                     rewards + self.discount_factor * self.get_value(next_states))
        return weights * torch.pow(predicted_q_values - temporal_difference_target, 2)

    def get_best_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.online_network(state)
            selected_action = torch.argmax(q_values)
        return selected_action.item()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.num_actions))
        return self.get_best_action(state)


class DuellingDQN:
    def __init__(self, epsilon, discount_factor, state_dimension, num_actions,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE, use_cuda=data_hyperparameters.USE_CUDA):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.state_dimension = state_dimension
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.online_feature_network = torch.nn.Sequential(torch.nn.Linear(state_dimension, hidden_size),
                                                          torch.nn.ReLU())
        self.online_advantage_network = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                            torch.nn.Linear(hidden_size, num_actions))
        self.online_value_network = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                        torch.nn.Linear(hidden_size, 1))
        self.target_feature_network = torch.nn.Sequential(torch.nn.Linear(state_dimension, hidden_size),
                                                          torch.nn.ReLU())
        self.target_advantage_network = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                            torch.nn.Linear(hidden_size, num_actions))
        self.target_value_network = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
                                                        torch.nn.Linear(hidden_size, 1))
        if use_cuda:
            self.online_feature_network.cuda()
            self.online_advantage_network.cuda()
            self.online_value_network.cuda()
            self.target_feature_network.cuda()
            self.target_advantage_network.cuda()
            self.target_value_network.cuda()
        self.synchronise()

    def synchronise(self):
        self.target_feature_network.load_state_dict(self.online_feature_network.state_dict())
        self.target_advantage_network.load_state_dict(self.online_advantage_network.state_dict())
        self.target_value_network.load_state_dict(self.online_value_network.state_dict())

    def adjust_epsilon(self, epsilon_decay_factor=data_hyperparameters.EPSILON_DECAY_FACTOR):
        self.epsilon *= epsilon_decay_factor

    def get_value(self, states):
        with torch.no_grad():
            online_features = self.online_feature_network(states)
            online_values = self.online_value_network(online_features)
            online_advantage = self.online_advantage_network(online_features)
            online_advantage_mean = torch.mean(online_advantage, dim=1, keepdim=True)
            predicted_q_values = online_values + online_advantage - online_advantage_mean
            selected_actions = torch.argmax(predicted_q_values, dim=1)
            target_features = self.target_feature_network(states)
            target_values = self.target_value_network(target_features)
            target_advantage = self.target_advantage_network(target_features)
            target_advantage_mean = torch.mean(target_advantage, dim=1, keepdim=True)
            target_q_values = target_values + target_advantage - target_advantage_mean
            batch_size = states.shape[0]
            values = target_q_values[range(batch_size), selected_actions]
        return values

    def calculate_temporal_difference_loss(self, states, actions, rewards, next_states, dones, weights):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            weights = torch.ones_like(rewards, dtype=torch.float32, device=self.device)
        online_features = self.online_feature_network(states)
        online_values = self.online_value_network(online_features)
        online_advantage = self.online_advantage_network(online_features)
        online_advantage_mean = torch.mean(online_advantage, dim=1, keepdim=True)
        predicted_q_values = online_values + online_advantage - online_advantage_mean
        batch_size = states.shape[0]
        predicted_q_values = predicted_q_values[range(batch_size), actions]
        with torch.no_grad():
            temporal_difference_target = torch.where(dones, rewards,
                                                     rewards + self.discount_factor * self.get_value(next_states))
        return weights * torch.pow(predicted_q_values - temporal_difference_target, 2)

    def get_best_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            online_features = self.online_feature_network(state)
            online_values = self.online_value_network(online_features)
            online_advantage = self.online_advantage_network(online_features)
            online_advantage_mean = torch.mean(online_advantage, dim=0, keepdim=True)
            q_values = online_values + online_advantage - online_advantage_mean
            selected_action = torch.argmax(q_values)
        return selected_action.item()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.num_actions))
        return self.get_best_action(state)


class TDLearner:
    def __init__(self, epsilon, discount_factor, num_actions, network, use_cuda=False):
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.network = network
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        if self.use_cuda:
            self.network.cuda()

    def get_value(self, state):
        raise NotImplementedError('This class method should not be directly called')

    def compute_temporal_difference_loss(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        batch_size = states.shape[0]
        predicted_q_values = self.network(states)[range(batch_size), actions]
        with torch.no_grad():
            predicted_next_state_values = self.get_value(next_states)
            target_q_values = torch.where(dones, rewards, rewards + self.discount_factor * predicted_next_state_values)
        loss = torch.mean((predicted_q_values - target_q_values) ** 2)
        return loss

    def get_best_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_values = self.network(state)
        if self.use_cuda:
            action_values = action_values.cpu()
        action_values = action_values.numpy()
        return argmax(action_values)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.num_actions))
        return self.get_best_action(state)


class QLearner(TDLearner):
    def get_value(self, states):
        with torch.no_grad():
            action_values = self.network(states)
        return torch.max(action_values, dim=1)[0]


class EVSARSALearner(TDLearner):
    def get_value(self, states):
        with torch.no_grad():
            action_values = self.network(states)
        max_actions = torch.argmax(action_values, dim=1)[0]
        return self.epsilon * torch.sum(action_values, dim=1) / self.num_actions \
               + (1 - self.epsilon) * action_values[:, max_actions]


class ActorAndCritic:
    def __init__(self, actor, critic, discount_factor, num_actions,
                 entropy_coefficient=data_hyperparameters.ENTROPY_COEFFICIENT, use_cuda=data_hyperparameters.USE_CUDA):
        self.actor = actor
        self.critic = critic
        self.discount_factor = discount_factor
        self.num_actions = num_actions
        self.entropy_coefficient = entropy_coefficient
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

    def compute_temporal_difference(self, states, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            next_states_value = self.critic(next_states)
        returns = torch.where(dones, rewards, rewards + self.discount_factor * next_states_value)
        return self.critic(states) - returns

    def compute_critic_loss(self, states, rewards, next_states, dones):
        return torch.pow(self.compute_temporal_difference(states, rewards, next_states, dones), 2)

    def compute_actor_loss(self, states, actions, rewards, next_states, dones, timesteps):
        with torch.no_grad():
            temporal_difference = self.compute_temporal_difference(states, rewards, next_states, dones)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        timesteps = torch.tensor(timesteps, dtype=torch.float32, device=self.device)
        state_action_preferences = self.actor(states)
        log_probabilities = torch.log_softmax(state_action_preferences, dim=-1)
        value = torch.mean(
            torch.pow(self.discount_factor, timesteps) * temporal_difference * log_probabilities[:, actions])
        if self.entropy_coefficient > 0.:
            probabilities = torch.softmax(state_action_preferences, dim=-1)
            neg_entropy = torch.sum(probabilities * log_probabilities, dim=-1)
            return value + self.entropy_coefficient * neg_entropy
        else:
            return value

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.actor(state), dim=0)
        return torch.multinomial(probabilities, num_samples=1).item()
