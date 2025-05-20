#!/usr/bin/env python
# ./batch_train_and_run.sh --overwrite

import gymnasium as gym
import os
import argparse
import torch
import numpy as np
import itertools 
import h5py 
import agent_class as agent

parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str, default='my_agent',
        help='input/output filename (suffix will be added by script)')
parser.add_argument('--N',type=int, default=1000,
        help='number of simulations')
parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)
parser.add_argument('--overwrite', action='store_true')
parser.set_defaults(overwrite=False)
parser.add_argument('--dqn', action='store_true')
                                                # via deep Q-learning
parser.add_argument('--ddqn', action='store_true') # use this flag to train 
                                                # via double deep Q-learning
parser.set_defaults(dqn=False)
parser.set_defaults(ddqn=False)
args = parser.parse_args()

# Create input and output filenames
input_filename = 'my_agent.tar'.format(args.f)
output_filename = '{0}_trajs.tar'.format(args.f)
N = args.N
verbose=args.verbose
overwrite=args.overwrite
dqn=args.dqn
ddqn=args.ddqn
if ddqn:
    dqn = True

if not overwrite:
    error_msg = ("File {0} already exists. If you want to overwrite"
        " that file, please restart the script with the flag --overwrite.")
    if os.path.exists(output_filename):
            raise RuntimeError(error_msg.format(output_filename))

def run_and_save_simulations(env, # environment
                            input_filename,output_filename,N=1000,
                            dqn=False):

    input_dictionary = torch.load(open(input_filename,'rb'), weights_only=False)
    dict_keys = np.array(list(input_dictionary.keys())).astype(int)
    max_index = np.max(dict_keys)
    input_dictionary = input_dictionary[max_index]
    parameters = input_dictionary['parameters']
    if dqn:
        my_agent = agent.dqn(parameters=parameters)
    else:
        my_agent = agent.actor_critic(parameters=parameters)
    my_agent.load_state(state=input_dictionary)
    #
    # instantiate environment
    env = gym.make('LunarLander-v3')
    #
    durations = []
    returns = []
    status_string = ("Run {0} of {1} completed with return {2:<5.1f}. Mean "
            "return over all episodes so far = {3:<6.1f}            ")
    # run simulations
    for i in range(N):
        # reset environment, duration, and reward
        state, info = env.reset()
        episode_return = 0.
        #
        for n in itertools.count():
            #
            action = my_agent.act(state)
            #
            state, step_reward, terminated, truncated, info = env.step(action)
            #
            done = terminated or truncated
            episode_return += step_reward
            #
            if done:
                #
                durations.append(n+1)
                returns.append(episode_return)
                #
                if verbose:
                    if i < N-1:
                        end ='\r'
                    else:
                        end = '\n'
                    print(status_string.format(i+1,N,episode_return,
                                        np.mean(np.array(returns))),
                                    end=end)
                break
    #
    dictionary = {'returns':np.array(returns),
                'durations':np.array(durations),
                'input_file':input_filename,
                'N':N}
        
    with h5py.File(output_filename, 'w') as hf:
        for key, value in dictionary.items():
            hf.create_dataset(str(key), 
                data=value)
    

# Create environment
env = gym.make('LunarLander-v3')

run_and_save_simulations(env=env,
                            input_filename=input_filename,
                            output_filename=output_filename,
                            N=N,
                            dqn=dqn)
