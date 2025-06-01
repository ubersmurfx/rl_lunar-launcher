#!/usr/bin/env python

import argparse
import os
import time
import gymnasium as gym
import agent_class as agent
from gymnasium import spaces
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='my_agent',
                    help='output filename (suffix will be added by script)')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--dqn', action='store_true') # use this flag to train 
                                                # via deep Q-learning
parser.add_argument('--ddqn', action='store_true') # use this flag to train 
                                                # via double deep Q-learning
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()

# Create output filenames
output_filename = '{0}.tar'.format(args.f)
output_filename_training_data = '{0}_training_data.h5'.format(args.f)
output_filename_time = '{0}_execution_time.txt'.format(args.f)
verbose=args.verbose
overwrite=args.overwrite
dqn=args.dqn
ddqn=args.ddqn

if not overwrite:
    # Comment the following out if you want to overwrite
    # existing model/training data
    error_msg = ("File {0} already exists. If you want to overwrite"
        " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(output_filename):
        raise RuntimeError(error_msg.format(output_filename))
    if os.path.exists(output_filename_training_data):
        raise RuntimeError(error_msg.format(output_filename_training_data))

#  Определяем обертку среды для изменения награды
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_params):
        super().__init__(env)
        self.reward_params = reward_params  #сохраняем параметры для расчета награды

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Модифицируем награду
        reward = self.custom_reward(observation, reward, terminated, truncated, info)

        return observation, reward, terminated, truncated, info

    def custom_reward(self, observation, reward, terminated, truncated, info):
        """
        Здесь определяем свою функцию награды.  Пример:

        * Дополнительная награда за близость к посадочной площадке.
        * Штраф за большое отклонение от вертикали.
        * Награда за мягкую посадку.
        """
        # Получаем координаты аппарата
        x, y,  vx, vy, angle, _, _, _ = observation

        # Награда за близость к центру (предположим, что центр в x=0)
        distance_reward = -abs(x) * self.reward_params['distance_penalty']  #отрицательная награда уменьшается при приближении к 0

        # Штраф за угол наклона
        angle_penalty = -abs(angle) * self.reward_params['angle_penalty']

        # Награда за скорость
        velocity_reward = -(abs(vx) + abs(vy)) * self.reward_params['velocity_penalty']

        #Добавляем к стандартной награде
        reward += distance_reward + angle_penalty + velocity_reward

        # Увеличиваем награду при посадке
        if terminated and reward > 0:
            reward += self.reward_params['landing_bonus']

        return reward

# Создаем параметры для модификации награды (их можно настроить)
reward_params = {
    'distance_penalty': 0.1,  # Насколько сильно штрафовать за расстояние от центра
    'angle_penalty': 0.1,   # Насколько сильно штрафовать за угол
    'velocity_penalty': 0.01, # Насколько сильно штрафовать за скорость
    'landing_bonus': 100.0 # Награда за посадку
}

env = gym.make('LunarLander-v3')

# Оборачиваем среду
env = CustomRewardWrapper(env, reward_params) # передаем параметры награды в обертку

N_actions = env.action_space.n
observation, info = env.reset()
N_state = len(observation)
if verbose:
    print('dimension of state space =',N_state)
    print('number of actions =',N_actions)

parameters = {
    'N_state':N_state,
    'N_actions':N_actions,
    'discount_factor':0.99,
    'N_memory':20000,
    'training_stride':5,
    'batch_size':32,
    'saving_stride':100,
    'n_episodes_max':10000,
    'n_solving_episodes':20,
    'solving_threshold_min':200.,
    'solving_threshold_mean':230.,
        }

if dqn or ddqn:
    if ddqn:
        parameters['doubledqn'] = True
    my_agent = agent.dqn(parameters=parameters)

# Train agent on environment
start_time = time.time()
training_results = my_agent.train(
                        environment=env,
                        verbose=verbose,
                        model_filename=output_filename,
                        training_filename=output_filename_training_data,
                            )
execution_time = (time.time() - start_time)
with open(output_filename_time,'w') as f:
    f.write(str(execution_time))

if verbose:
    print('Execution time in seconds: ' + str(execution_time))
