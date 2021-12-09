# -*- coding: utf-8 -*-
"""
The the graphs are generated using this script

A conversion from Jupyter Notebook
Source-code copyright: quwsarohi@gmail.com
"""

# Commented out IPython magic to ensure Python compatibility.

import sys
import numpy as np
import os
import pickle
import time
from math import ceil
import random
from glob import glob
 
import sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 

import tensorflow as tf
import tensorflow.keras.backend as K
from .agent_environment import *

import seaborn as sns
sns.set_style("whitegrid")


# --------------------------- Agent Setup --------------------------- 

print('Loading agent weights')

agent = DQN(input_shape=(30, 7), action_size=3, batch_size=128, UPDATE_TARGET=4,
            DISCOUNT=0.9, DISCOUNT_DECAY=0, timestep=31, REPLAY_MEM_SZ=2000) 

model_path = os.path.join(os.path.realpath('.'), 'base_model')
if os.path.exists(model_path) == False:
    print('Trained model not found')

agent.model = tf.keras.models.load_model(model_path)
 
#1000 = 224 x 224
#5000 = 500 x 500
#10000 = 708 x 708
# population_density = 0.02

print('Starting Environment')

# Modify the seed_list to generate different random-environmental states
env_actions = ['no-lockdown', 'semi-lockdown', 'lockdown', 'agent']
envSetups = [{'height':577 , 'population':10000, 'seed':1, 'infected':70},
            {'height':708 , 'population':10000, 'seed':6, 'infected':70},
            {'height':1000 , 'population':10000, 'seed':4, 'infected':70}]


for envSetup in envSetups:
    for act in env_actions:
        print('Starting environ with the parameters', envSetup)
        print('The actions are based on', act)

        env = Env(height=envSetup['height'], width=envSetup['height'], 
                  population=envSetup['population'], 
                  infected_ratio=0.002, beta=np.inf, prob=1, boundary=0, 
                  cure_after=21, infect_after=2, death_distribution=[0.2, 0.8], 
                  day_step=15, day_limit=0, normalize=True, linear_reward=False)

        done = False
        tot_reward, actions, real_act = 0, 0, 0
        st = time.time()
        data = {'Day': [], 'R0': [], 'Active Cases': [], 'Infected': [], 
                'Economy': [], 'Death': [], 'Cured': [], 
                'Action': [], 'Population': env.population, 'R00': [], 
                'Reward': [], 'Action1': [], 'Action2': [], 'Action3': []} 

        agent.initReplay()

        # Manually setting the number of infected population
        env.first_infected = envSetup['infected']

        current_state = np.array(list(env.reset(envSetup['seed']))+[0], dtype=np.float32)
        cycle_count = 0
        best_action = 0

        while not done:
            if act == 'agent':
                reward_preds = agent.get_qs(current_state)
                best_action = np.argmax(reward_preds)
                data['Action1'].append(reward_preds[0, 0])
                data['Action2'].append(reward_preds[0, 1])
                data['Action3'].append(reward_preds[0, 2])
            if act == 'lockdown':
                best_action = 2
            if act == 'semi-lockdown':
                best_action = 1
            if act == 'no-lockdown':
                best_action = 0
            
            data['Day'].append(env.log['day'])
            data['Active Cases'].append(env.log['active_cases'])
            data['Infected'].append(env.log['infected'])
            data['Cured'].append(env.log['cured'])
            data['Economy'].append(env.log['economy'])
            data['R0'].append(env.log['R0'])
            data['R00'].append(env.log['R00'])
            data['Death'].append(env.log['death'])
            data['Action'].append(best_action)
    
            # Perform the action
            new_state, reward, done, dct = env.step(best_action)
            new_state = np.array(list(new_state)+[best_action], dtype=np.float32)
            data['Reward'].append(reward)
    
            # Find the q values of the new state (after performing previous action)
            tot_reward += reward
    
            agent.update_replay_memory((current_state, best_action, reward, new_state, done))
            #agent.train(done, actions)
            current_state = new_state
            print(f"\rDay:{data['Day'][-1]}, R0:{data['R0'][-1]:0.1f}, Infected:{data['Infected'][-1]}, AC:{data['Active Cases'][-1]}", end='')
        
        rl = len(data['Reward'])
        data['RewardSum'] = [x for x in data['Reward']]
        for i in range(rl-2, -1, -1):
            data['RewardSum'][i] += data['Reward'][i+1]*agent.DISCOUNT
        data['Total Infected'] = env.total_infected
        data['Total Cured'] = env.total_cured
        data['Total Death'] = env.total_death
        data['Total Economy'] = np.sum(data['Economy'])
        data['MAX_ECONOMY'] = env.observation_space_high[2]
        data['TOTAL_POPULATION'] = env.observation_space_high[1]
        
        draw(data, rewards=False, plot=False, 
            #title=f"Population: {env.population}, Initial Infectious: {infpop}",
            filename=os.path.join(os.path.realpath('.'), 
                                  "dashboard_logs", 
                                  f"{env.height}_{env.population}_{envSetup['seed']}_{envSetup['infected']}_{act}")
            )
        print('Log saved in dashboard_logs')
        #actionvals.append(np.sum(data['Reward']))
