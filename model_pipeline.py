import torch
import data_hyperparameters
from datetime import datetime
import os
import csv
from log_utils import create_logger, write_log
from pickle import dump, load
import pandas as pd
import numpy as np
import model_classes

LOG_FILE = 'model_pipeline'
logger = create_logger(LOG_FILE)

if not os.path.exists('saved_models/'):
    os.mkdir('saved_models/')


def moving_average(x, span=100):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values


def play_and_train_network(env, agent, actor_optimiser, critic_optimiser, max_episode_length=500):
    total_reward = 0.
    s = env.reset()
    for t in range(max_episode_length):
        a = agent.get_action(s)
        next_s, r, done, _ = env.step(a)
        critic_optimiser.zero_grad()
        agent.compute_critic_loss([s], [r], [next_s], [done]).backward()
        critic_optimiser.step()
        actor_optimiser.zero_grad()
        agent.compute_actor_loss([s], [a], [r], [next_s], [done], [t]).backward()
        actor_optimiser.step()
        s = next_s
        total_reward += r
        if done:
            break
    return total_reward


def run_experiment(env, agent, name, log_object, episodes, episodes_for_logging, lr=1e-4):
    actor_optimiser = torch.optim.Adam(agent.actor.parameters(), lr=lr)
    critic_optimiser = torch.optim.Adam(agent.critic.parameters(), lr=lr)
    rewards = []
    now_for_episode = datetime.now()
    for episode in range(episodes):
        rewards.append(play_and_train_network(env, agent, actor_optimiser, critic_optimiser))
        if episode % episodes_for_logging == 0 and episode != 0:
            write_log('Computed {0} out of {1} episodes for {2} in {3} seconds'.format(episode, episodes, name, (datetime.now() - now_for_episode).total_seconds()), log_object)
            now_for_episode = datetime.now()
            write_log('Mean reward: {0}'.format(np.mean(rewards[-episodes_for_logging:])), log_object)
    return rewards


def train_agent(epsilon, discount_factor, env, agent_name, log_object, dqn_type='duelling', replay_type='prioritised'):
    write_log('Running agent {0}'.format(agent_name), log_object)
    now = datetime.now()
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    if dqn_type == 'duelling':
        agent = model_classes.DuellingDQN(epsilon, discount_factor, state_dim, n_actions)
        optimiser = torch.optim.Adam(
            [{'params': agent.online_feature_network.parameters()}, {'params': agent.online_value_network.parameters()},
             {'params': agent.online_advantage_network.parameters()}])
    else:
        agent = model_classes.DoubleDQN(epsilon, discount_factor, state_dim, n_actions)
        optimiser = torch.optim.Adam(agent.online_network.parameters())
    if replay_type == 'prioritised':
        replay = model_classes.PrioritisedReplay(data_hyperparameters.REPLAY_CAPACITY)
    else:
        replay = model_classes.Replay(data_hyperparameters.REPLAY_CAPACITY)
    s = env.reset()
    total_reward = 0.
    total_rewards = []
    for t in range(data_hyperparameters.NUM_STEPS):
        a = agent.get_action(s)
        next_s, r, done, _ = env.step(a)
        total_reward += r
        replay.add(s, a, r, next_s, done)
        if len(replay) >= data_hyperparameters.REPLAY_BATCH_SIZE:
            s_batch, a_batch, r_batch, next_s_batch, done_batch, i_batch, w_batch = replay.sample(data_hyperparameters.REPLAY_BATCH_SIZE)
            batch_loss = agent.calculate_temporal_difference_loss(s_batch, a_batch, r_batch, next_s_batch, done_batch, w_batch)
            if replay_type == 'prioritised':
                with torch.no_grad():
                    new_priorities = batch_loss + data_hyperparameters.PRIORITY_ADJUSTMENT_EPSILON
                    if agent.use_cuda:
                        new_priorities = new_priorities.cpu()
                    new_priorities = new_priorities.numpy()
                replay.update_priorities(i_batch, new_priorities)
            optimiser.zero_grad()
            batch_loss = torch.mean(batch_loss)
            batch_loss.backward()
            optimiser.step()
        s = next_s
        agent.adjust_epsilon()
        replay.adjust_beta()
        if done:
            total_rewards.append(total_reward)
            total_reward = 0.
            s = env.reset()
        if t % data_hyperparameters.NUM_STEPS_FOR_SYNCHRONISING == 0:
            agent.synchronise()
    write_log('Agent {0} took {1} seconds total'.format(agent_name, (datetime.now() - now).total_seconds()), log_object)
    return total_rewards
