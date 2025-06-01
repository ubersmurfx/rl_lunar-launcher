#!/usr/bin/env python

import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
device = torch.device("cpu") 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'
import warnings
from torch.distributions import Categorical

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class memory(): # class for experience replay buffer для value-based метода DQN
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity) # максимальный размер буфера

    def push(self, *args): # добавление нового опыта в бфер
        self.memory.append(Transition(*args))

    def sample(self, batch_size): # выбирает случаный батч из буфера опыта
        return random.sample(self.memory, batch_size)

    def __len__(self): # возвращает текущий размер буфера опыта
        return len(self.memory)

class neural_network(nn.Module): # класс для полносвязной нейронной сети
    def __init__(self,
                layers=[8,64,32,4],
                dropout=False,
                p_dropout=0.5, # используем такой метод регуляризации
                ):
        super(neural_network,self).__init__()

        self.network_layers = []
        n_layers = len(layers)
        for i,neurons_in_current_layer in enumerate(layers[:-1]):
            self.network_layers.append(nn.Linear(neurons_in_current_layer, 
                                                layers[i+1]) )
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )
            if i < n_layers - 2:
                self.network_layers.append( nn.ReLU() )
        self.network_layers = nn.Sequential(*self.network_layers)

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x

class agent_base():
    def __init__(self,parameters):
        parameters = self.make_dictionary_keys_lowercase(parameters)
        self.set_initialization_parameters(parameters=parameters)
        default_parameters = self.get_default_parameters()
        parameters = self.merge_dictionaries(dict1=parameters,
                                             dict2=default_parameters)
        self.set_parameters(parameters=parameters)
        self.parameters = copy.deepcopy(parameters)
        self.initialize_neural_networks(neural_networks=\
                                            parameters['neural_networks'])
        self.initialize_optimizers(optimizers=parameters['optimizers'])
        self.initialize_losses(losses=parameters['losses'])
        self.in_training = False        #

    def make_dictionary_keys_lowercase(self,dictionary):
        output_dictionary = {}
        for key, value in dictionary.items():
            output_dictionary[key.lower()] = value
        return output_dictionary

    def merge_dictionaries(self,dict1,dict2):
        return_dict = copy.deepcopy(dict1)
        dict1_keys = return_dict.keys()
        for key, value in dict2.items():
            if key not in dict1_keys:
                return_dict[key] = value
        return return_dict

    def get_default_parameters(self):
        parameters = {
            'neural_networks':
                {
                'policy_net':{
                    'layers':[self.n_state,128,32,self.n_actions],
                            }
                },
            'optimizers':
                {
                'policy_net':{
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3}, # learning rate
                            }
                },
            'losses':
                {
                'policy_net':{            
                    'loss':'MSELoss',
                }
                },
            'n_memory':20000, # попробовать поменять
            'training_stride':5,
            'batch_size':32,
            'saving_stride':100,
            'n_episodes_max':10000,
            'n_solving_episodes':20,
            'solving_threshold_min':200,
            'solving_threshold_mean':230,
            'discount_factor':0.99,
            }
        parameters = self.make_dictionary_keys_lowercase(parameters)
        return parameters

    def set_initialization_parameters(self,parameters):
        try:
            self.n_state = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state (= # of input"\
                         +" nodes for neural network) needs to be supplied.")
        try:
            self.n_actions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions (= # of output"\
                         +" nodes for neural network) needs to be supplied.")

    def set_parameters(self,parameters):
        parameters = self.make_dictionary_keys_lowercase(parameters)
        try:
            self.discount_factor = parameters['discount_factor']
        except KeyError:
            pass
        try:
            self.n_memory = int(parameters['n_memory'])
            self.memory = memory(self.n_memory)
        except KeyError:
            pass
        try:
            self.training_stride = parameters['training_stride']
        except KeyError:
            pass
        try:
            self.batch_size = int(parameters['batch_size'])
        except KeyError:
            pass
        try:
            self.saving_stride = parameters['saving_stride']
        except KeyError:
            pass
        try:
            self.n_episodes_max = parameters['n_episodes_max']
        except KeyError:
            pass
        try:
            self.n_solving_episodes = parameters['n_solving_episodes']
        except KeyError:
            pass
        try:
            self.solving_threshold_min = parameters['solving_threshold_min']
        except KeyError:
            pass
        try:
            self.solving_threshold_mean = parameters['solving_threshold_mean']
        except KeyError:
            pass

    def get_parameters(self): # getter
        return self.parameters

    def initialize_neural_networks(self,neural_networks):
        self.neural_networks = {}
        for key, value in neural_networks.items():
            self.neural_networks[key] = neural_network(value['layers']).to(device)
        
    def initialize_optimizers(self,optimizers):
        self.optimizers = {}
        for key, value in optimizers.items():
            self.optimizers[key] = torch.optim.RMSprop(
                        self.neural_networks[key].parameters(),
                            **value['optimizer_args'])
    
    def initialize_losses(self,losses):
        self.losses = {}
        for key, value in losses.items():
            self.losses[key] = nn.MSELoss()

    def get_number_of_model_parameters(self,name='policy_net'): 
        return sum(p.numel() for p in self.neural_networks[name].parameters() \
                                    if p.requires_grad)

    def get_state(self):
        state = {'parameters':self.get_parameters()}
        for name,neural_network in self.neural_networks.items():
            state[name] = copy.deepcopy(neural_network.state_dict())
        for name,optimizer in (self.optimizers).items():
            state[name+'_optimizer'] = copy.deepcopy(optimizer.state_dict())
        return state
    
    def load_state(self,state):
        parameters=state['parameters']
        self.check_parameter_dictionary_compatibility(parameters=parameters)
        self.__init__(parameters=parameters)
        for name,state_dict in (state).items():
            if name == 'parameters':
                continue
            elif 'optimizer' in name:
                name = name.replace('_optimizer','')
                self.optimizers[name].load_state_dict(state_dict)
            else:
                self.neural_networks[name].load_state_dict(state_dict)

    def check_parameter_dictionary_compatibility(self,parameters):
        error_string = ("Error loading state. Provided parameter {0} = {1} ",
                    "is inconsistent with agent class parameter {0} = {2}. ",
                    "Please instantiate a new agent class with parameters",
                    " matching those of the model you would like to load.")
        try: 
            n_state =  parameters['n_state']
            if n_state != self.n_state:
                raise RuntimeError(error_string.format('n_state',n_state,
                                                self.n_state))
        except KeyError:
            pass
        try: 
            n_actions =  parameters['n_actions']
            if n_actions != self.n_actions:
                raise RuntimeError(error_string.format('n_actions',n_actions,
                                                self.n_actions))
        except KeyError:
            pass

    def evaluate_stopping_criterion(self,list_of_returns):
        if len(list_of_returns) < self.n_solving_episodes:
            return False, 0., 0.
        recent_returns = np.array(list_of_returns)
        recent_returns = recent_returns[-self.n_solving_episodes:]
        minimal_return = np.min(recent_returns)
        mean_return = np.mean(recent_returns)
        if minimal_return > self.solving_threshold_min:
            if mean_return > self.solving_threshold_mean:
                return True, minimal_return, mean_return
        return False, minimal_return, mean_return

    def act(self,state):
        return np.random.randint(self.n_actions) 


    def add_memory(self,memory):
        self.memory.push(*memory)

    def get_samples_from_memory(self):
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*current_transitions))
        state_batch = torch.cat( [s.unsqueeze(0) for s in batch.state],
                                        dim=0)#.to(device)
        next_state_batch = torch.cat(
                         [s.unsqueeze(0) for s in batch.next_state],dim=0)
        action_batch = torch.cat(batch.action)#.to(device)
        reward_batch = torch.cat(batch.reward)#.to(device)
        done_batch = torch.tensor(batch.done).float()#.to(device)
        return state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch

    def run_optimization_step(self, epoch):
        """Run one optimization step
        
        Keyword argument:
        epoch (int) -- number of current training epoch
        """
        #
        # Here is where the actual optimization happens.
        # 
        # This method MUST be implemented in any child class, and might look
        # very different depending on the learning algorithm. 
        # Note that any implementation must contain the argument "epoch", as 
        # this method is called as run_optimization_step(epoch=epoch) in the
        # method self.train() below.
        # 
        # For examples see the child classes "dqn" and "actor_critic" below
        #
        
    

    def train(self,environment,
                    verbose=True,
                    model_filename=None,
                    training_filename=None,
                ):
        self.in_training = True
        training_complete = False
        step_counter = 0
        epoch_counter = 0
        episode_durations = []
        episode_returns = []
        steps_simulated = []
        training_epochs = []
        output_state_dicts = {}
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            status_progress_string = (
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        for n_episode in range(self.n_episodes_max):
            state, info = environment.reset()
            current_total_reward = 0.
            for i in itertools.count():
                action = self.act(state=state)
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)
                step_counter += 1
                done = terminated or truncated
                current_total_reward += reward
                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                self.add_memory([torch.tensor(state),
                            action,
                            torch.tensor(next_state),
                            reward,
                            done])
                state = next_state
                if step_counter % self.training_stride == 0:
                    self.run_optimization_step(epoch=epoch_counter)
                    epoch_counter += 1
                if done:
                    episode_durations.append(i + 1)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)
                    training_complete, min_ret, mean_ret = \
                            self.evaluate_stopping_criterion(\
                                list_of_returns=episode_returns)
                    if verbose:
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if min_ret > self.solving_threshold_min:
                                if mean_ret > self.solving_threshold_mean:
                                    end='\n'
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   min_ret,mean_ret),
                                        end=end)
                    break
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.n_episodes_max-1:
                if model_filename != None:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)
                training_results = {'episode_durations':episode_durations,
                            'epsiode_returns':episode_returns,
                            'n_training_epochs':training_epochs,
                            'n_steps_simulated':steps_simulated,
                            'training_completed':False,
                            }
                if training_filename != None:
                    self.save_dictionary(dictionary=training_results,
                                        filename=training_filename)
            if training_complete:
                training_results['training_completed'] = True
                break
        if not training_complete:
            warning_string = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warning_string.format(self.n_episodes_max))
        self.in_training = False
        return training_results

    def save_dictionary(self,dictionary,filename):
        """Save a dictionary in hdf5 format"""

        with h5py.File(filename, 'w') as hf:
            self.save_dictionary_recursively(h5file=hf,
                                            path='/',
                                            dictionary=dictionary)
                
    def save_dictionary_recursively(self,h5file,path,dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.save_dictionary_recursively(h5file, 
                                                path + str(key) + '/',
                                                value)
            else:
                h5file[path + str(key)] = value

    def load_dictionary(self,filename):
        with h5py.File(filename, 'r') as hf:
            return self.load_dictionary_recursively(h5file=hf,
                                                    path='/')

    def load_dictionary_recursively(self,h5file, path):
        return_dict = {}
        for key, value in h5file[path].items():
            if isinstance(value, h5py._hl.dataset.Dataset):
                return_dict[key] = value.value
            elif isinstance(value, h5py._hl.group.Group):
                return_dict[key] = self.load_dictionary_recursively(\
                                            h5file=h5file, 
                                            path=path + key + '/')
        return return_dict

class dqn(agent_base):
    def __init__(self,parameters):
        super().__init__(parameters=parameters)
        self.in_training = False

    def get_default_parameters(self):
        default_parameters = super().get_default_parameters()
        default_parameters['neural_networks']['target_net'] = {}
        default_parameters['neural_networks']['target_net']['layers'] = \
        copy.deepcopy(\
                default_parameters['neural_networks']['policy_net']['layers'])
        default_parameters['target_net_update_stride'] = 1 
        default_parameters['target_net_update_tau'] = 1e-2 
        default_parameters['epsilon'] = 1.0
        default_parameters['epsilon_1'] = 0.1
        default_parameters['d_epsilon'] = 0.00005
        default_parameters['doubledqn'] = False
        return default_parameters


    def set_parameters(self,parameters):
        super().set_parameters(parameters=parameters)
        try:
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        try:
            self.target_net_update_stride = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass
        try:
            self.target_net_update_tau = parameters['target_net_update_tau']
            error_msg = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            error_msg = error_msg.format(self.target_net_update_tau)
            if self.target_net_update_tau < 0:
                raise RuntimeError(error_msg)
            elif self.target_net_update_tau > 1:
                raise RuntimeError(error_msg)
        except KeyError:
            pass
        try:
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        try:
            self.epsilon_1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass
        try:
            self.d_epsilon = \
                    parameters['d_epsilon']
        except KeyError:
            pass

    def act(self,state,epsilon=0.):
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            policy_net = self.neural_networks['policy_net']
            with torch.no_grad():
                policy_net.eval()
                action = policy_net(torch.tensor(state)).argmax(0).item()
                policy_net.train()
                return action
        else:
            return torch.randint(low=0,high=self.n_actions,size=(1,)).item()
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self,epoch):

        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch = self.get_samples_from_memory()
        policy_net = self.neural_networks['policy_net']
        target_net = self.neural_networks['target_net']
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net']
        policy_net.train()
        LHS = policy_net(state_batch.to(device)).gather(dim=1,
                                 index=action_batch.unsqueeze(1))
        if self.doubleDQN:
            argmax_next_state = policy_net(next_state_batch).argmax(
                                                                    dim=1)
            Q_next_state = target_net(next_state_batch).gather(
                dim=1,index=argmax_next_state.unsqueeze(1)).squeeze(1)
        else:
            Q_next_state = target_net(next_state_batch\
                                                ).max(1)[0].detach()
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        RHS = RHS.unsqueeze(1)
        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        policy_net.eval()
        self.update_epsilon()
        if epoch % self.target_net_update_stride == 0:
            self.soft_update_target_net()
        
    def soft_update_target_net(self):
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.neural_networks['target_net'].load_state_dict(dict_params2)
