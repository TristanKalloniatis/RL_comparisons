import gym
import data_hyperparameters
from log_utils import create_logger, write_log
from datetime import datetime
import model_classes
import matplotlib.pyplot as plt
from model_pipeline import run_experiment, moving_average
import torch

LOG_FILE = 'run_experiment'
logger = create_logger(LOG_FILE)

cart_pole = gym.make('CartPole-v0')
n_actions = cart_pole.action_space.n
state_dim = cart_pole.observation_space.shape[0]

all_experiment_rewards = []
all_experiment_names = []

write_log('Environment: {0}'.format('CartPole-v0'), logger)
write_log('Hyperparameters', logger)
write_log('num_experiments: {0}'.format(data_hyperparameters.NUM_EXPERIMENTS), logger)
write_log('epsilon: {0}'.format(data_hyperparameters.EPSILON), logger)
write_log('discount_factor: {0}'.format(data_hyperparameters.DISCOUNT_FACTOR), logger)
write_log('epsilon_decay_factor: {0}'.format(data_hyperparameters.EPSILON_DECAY_FACTOR), logger)
write_log('num_episodes: {0}'.format(data_hyperparameters.NUM_EPISODES), logger)
write_log('num_episodes_for_decay: {0}'.format(data_hyperparameters.NUM_EPISODES_FOR_DECAY), logger)

for experiment in range(data_hyperparameters.NUM_EXPERIMENTS):
    experiment_name = 'QLearner_{0}'.format(experiment)
    all_experiment_names.append(experiment_name)
    write_log('Building agent for experiment {0}'.format(experiment_name), logger)
    now = datetime.now()
    q_agent = model_classes.QLearner(data_hyperparameters.EPSILON, data_hyperparameters.DISCOUNT_FACTOR, n_actions, torch.nn.Sequential(torch.nn.Linear(state_dim, data_hyperparameters.HIDDEN_SIZE), torch.nn.ReLU(),
                                                                       torch.nn.Linear(data_hyperparameters.HIDDEN_SIZE, data_hyperparameters.HIDDEN_SIZE),
                                                                       torch.nn.ReLU(),
                                                                       torch.nn.Linear(data_hyperparameters.HIDDEN_SIZE, n_actions)))
    all_experiment_rewards.append(run_experiment(cart_pole, q_agent, experiment_name, logger, data_hyperparameters.EPSILON_DECAY_FACTOR,
                                                 data_hyperparameters.NUM_EPISODES, data_hyperparameters.NUM_EPISODES_FOR_DECAY))
    write_log('{0} took {1} seconds in total'.format(experiment_name, (datetime.now() - now).total_seconds()), logger)
    experiment_name = 'EVSARSALearner_{0}'.format(experiment)
    all_experiment_names.append(experiment_name)
    write_log('Building agent for experiment {0}'.format(experiment_name), logger)
    now = datetime.now()
    evsarsa_agent = model_classes.EVSARSALearner(data_hyperparameters.EPSILON, data_hyperparameters.DISCOUNT_FACTOR, n_actions, torch.nn.Sequential(torch.nn.Linear(state_dim, data_hyperparameters.HIDDEN_SIZE),
                                                                                   torch.nn.ReLU(),
                                                                                   torch.nn.Linear(data_hyperparameters.HIDDEN_SIZE, data_hyperparameters.HIDDEN_SIZE),
                                                                                   torch.nn.ReLU(),
                                                                                   torch.nn.Linear(data_hyperparameters.HIDDEN_SIZE, n_actions)))
    all_experiment_rewards.append(run_experiment(cart_pole, evsarsa_agent, experiment_name, logger,
                                                 data_hyperparameters.EPSILON_DECAY_FACTOR, data_hyperparameters.NUM_EPISODES, data_hyperparameters.NUM_EPISODES_FOR_DECAY))
    write_log('{0} took {1} seconds in total'.format(experiment_name, (datetime.now() - now).total_seconds()), logger)

for experiment in range(len(all_experiment_names)):
    plt.plot(moving_average(all_experiment_rewards[experiment]), label=all_experiment_names[experiment])
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.grid()
plt.legend()
plt.show()
