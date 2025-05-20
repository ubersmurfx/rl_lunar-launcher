#!/usr/bin/env python

import argparse
import os
import time
import gymnasium as gym

import agent_class as agent

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

# Create environment
env = gym.make('LunarLander-v3')

# Obtain dimensions of action and observation space
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
    #
    my_agent = agent.dqn(parameters=parameters)
else:
    my_agent = agent.actor_critic(parameters=parameters)


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


