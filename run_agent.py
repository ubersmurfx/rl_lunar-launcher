#!/usr/bin/env python

import gymnasium as gym
import os
import argparse
import torch
import numpy as np
import itertools
import h5py
import agent_class as agent
from gymnasium import spaces


parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, default='my_agent',
                    help='input/output filename (suffix will be added by script)')
parser.add_argument('--N', type=int, default=1000,
                    help='number of simulations')
parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--overwrite', action='store_true')
parser.set_defaults(overwrite=False)
parser.add_argument('--dqn', action='store_true')
parser.add_argument('--ddqn', action='store_true')
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()

# Create input and output filenames
input_filename = '{0}.tar'.format(args.f)
output_filename = '{0}_trajs.tar'.format(args.f)
N = args.N
verbose = args.verbose
overwrite = args.overwrite
dqn = args.dqn
ddqn = args.ddqn
if ddqn:
    dqn = True

if not overwrite:
    # Comment the following out if you want to overwrite
    # existing model/training data
    error_msg = ("File {0} already exists. If you want to overwrite"
                 " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(output_filename):
        raise RuntimeError(error_msg.format(output_filename))


# Определяем обертку среды для изменения награды
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_params):
        super().__init__(env)
        self.reward_params = reward_params

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Модифицируем награду
        reward = self.custom_reward(observation, reward, terminated, truncated, info)

        return observation, reward, terminated, truncated, info

    def custom_reward(self, observation, reward, terminated, truncated, info):
        """
        Здесь определяем свою функцию награды.
        """
        x, y, vx, vy, angle, _, _, _ = observation

        # Награда за близость к центру
        distance_reward = -abs(x) * self.reward_params['distance_penalty']

        # Штраф за угол наклона
        angle_penalty = -abs(angle) * self.reward_params['angle_penalty']

        # Награда за скорость
        velocity_reward = -(abs(vx) + abs(vy)) * self.reward_params['velocity_penalty']

        reward += distance_reward + angle_penalty + velocity_reward

        if terminated and reward > 0:
            reward += self.reward_params['landing_bonus']

        return reward


def run_and_save_simulations(env,
                             input_filename, output_filename, N=1000,
                             dqn=False):
    input_dictionary = torch.load(open(input_filename, 'rb'))
    dict_keys = np.array(list(input_dictionary.keys())).astype(int)
    max_index = np.max(dict_keys)
    input_dictionary = input_dictionary[max_index]  # During training we
    parameters = input_dictionary['parameters']
    if dqn:
        my_agent = agent.dqn(parameters=parameters)
    else:
        my_agent = agent.actor_critic(parameters=parameters)
    my_agent.load_state(state=input_dictionary)
    durations = []
    returns = []
    status_string = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
                     "return over all episodes so far = {3:<6.1f}            ")
    for i in range(N):
        state, info = env.reset()
        episode_return = 0.
        for n in itertools.count():
            action = my_agent.act(state)
            state, step_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += step_reward
            if done:
                durations.append(n + 1)
                returns.append(episode_return)
                if verbose:
                    if i < N - 1:
                       end = '\r'
                    else:
                        end = '\n'
                    print(status_string.format(i + 1, N, episode_return,
                                                np.mean(np.array(returns))),
                          end=end)
                break
    dictionary = {'returns': np.array(returns),
                  'durations': np.array(durations),
                  'input_file': input_filename,
                  'N': N}

    with h5py.File(output_filename, 'w') as hf:
        for key, value in dictionary.items():
            hf.create_dataset(str(key),
                              data=value)


# Создаем параметры для модификации награды
reward_params = {
    'distance_penalty': 0.1,
    'angle_penalty': 0.1,
    'velocity_penalty': 0.01,
    'landing_bonus': 100.0
}

env = gym.make('LunarLander-v3')

# Оборачиваем среду
env = CustomRewardWrapper(env, reward_params)

run_and_save_simulations(env=env,
                             input_filename=input_filename,
                             output_filename=output_filename,
                             N=N,
                             dqn=dqn)
