# -*- coding: utf-8 -*-
"""
The agent is trained using this script

A conversion from Jupyter Notebook
Source-code copyright: quwsarohi@gmail.com
"""

# Commented out IPython magic to ensure Python compatibility.
import sys, os, warnings
import numpy as np
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4' 
import pickle
import time
from math import ceil
import random
from glob import glob
import tensorflow as tf
#tf.autograph.set_verbosity(4)
#tf.get_logger().setLevel('FATAL')
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import tensorflow.keras.backend as K
from agent_environment import *

#print('Tensorflow Logging:', tf.compat.v1.logging.get_verbosity())


# A function that saves the agent model
def saveFile(epi, sdir, saveIter=None):
    global epsilon
    global START_EPSILON_DECAYING
    global END_EPSILON_DECAYING
    global epsilon_decay_value
    global episode
 
    params = {'epsilon' : epsilon,'START_EPSILON_DECAYING' : START_EPSILON_DECAYING,
              'END_EPSILON_DECAYING' : END_EPSILON_DECAYING, 'episode' : epi,
              'epsilon_decay_value' : epsilon_decay_value, 
            }
    
    with open(sdir+'params.pkl', 'wb') as f:
        pickle.dump(params, f)
    
    if saveIter != None and epi%saveIter == 0:
        agent.model.save(sdir+f'agent_{epi}')
    
    agent.model.save(sdir+'model')
    agent.target_model.save(sdir+'tmodel')
    with open(sdir+f'replay.pkl', 'wb') as f:
        pickle.dump(agent.replay_memory, f)


# A function that loads the agent model 
def loadFile(sdir, loadParam=True, loadopt=False):
    files = glob(sdir+'*.pkl', recursive=False)
    if len(files) == 0:
        print('No files found!')
        return
    else:
        print('Files found:', [str(a).split('/')[-1] for a in files])
 
    global agent
    global epsilon
    global START_EPSILON_DECAYING
    global END_EPSILON_DECAYING
    global epsilon_decay_value
    global episode
 
    agent.model = tf.keras.models.load_model(sdir+'model')
    agent.target_model = tf.keras.models.load_model(sdir+'tmodel')

    with open(sdir+f'replay.pkl', 'wb') as f:
        pickle.dump(agent.replay_memory, f)
 
    for file in files:
        if 'params' in file and loadParam:
            with open(file, 'rb') as f:
                fp = pickle.load(f)
                epsilon = fp['epsilon']
                START_EPSILON_DECAYING = fp['START_EPSILON_DECAYING']
                END_EPSILON_DECAYING = fp['END_EPSILON_DECAYING']
                epsilon_decay_value = fp['epsilon_decay_value']
                episode = fp['episode']


# The DEF_SDIR is modified for security purpose
# Change this where the agent model is to be saved
DEF_SDIR = os.path.join(os.path.realpath('.'), 'saved_params', '')
 
#1000 = 224 x 224
#5000 = 500 x 500
#10000 = 708 x 708 

# Creating the virtual environment state
# An approximate value is initialized, default height x width is 708x708
env = Env(height=500, width=500, population_density=0.02, infected_ratio=0.003,
          beta=np.inf, prob=1, boundary=0, cure_after=21, infect_after=2, 
          death_distribution=[0.2, 0.8], day_step=15, day_limit=0, 
          normalize=True, linear_reward=False)
 
agent = DQN(input_shape=(30, 7), action_size=3, batch_size=128, UPDATE_TARGET=4,
            DISCOUNT=0.9, DISCOUNT_DECAY=0, timestep=31, REPLAY_MEM_SZ=2000)
 

############### Loading Params (Approx.)
# Make a random movement if the value is less than epsilon
# Exploration settings

EPISODES = 200      # Altered due to testing
epsilon = 1         # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
episode = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)
agentAVG = deque(maxlen=10)
randagentAVG = deque(maxlen=10)

# Load if previously trained file exists 
loadFile(DEF_SDIR, loadParam=False)


print(f'Training starting from episode {episode}, Epsilon {epsilon:.2f}')
 
for epi in range(episode, EPISODES):
    done = False
    tot_reward, actions, real_act = 0, 0, 0
    st = time.time()
    data = {'Day': [], 'R0': [], 'Active Cases': [], 'Infected': [], 
            'Economy': [], 'Death': [], 'Cured': [], 
            'Action': [], 'Population': env.population, 'R00': [], 
            'Reward': [], 'Action1': [], 'Action2': [], 'Action3': []} 
 
    env.first_infected = random.randint(20, 200)
    current_state = np.array(list(env.reset())+[0], dtype=np.float32)
 
    while not done:
        # In every 10th step, the agent performs all operation by itself
        # This is done to check the activity/knowledge of the agent
        if epi%10 == 0:
            # Select the best action
            real_act += 1
            reward_preds = agent.get_qs(current_state)
            best_action = np.argmax(reward_preds)
            data['Action1'].append(reward_preds[0, 0])
            data['Action2'].append(reward_preds[0, 1])
            data['Action3'].append(reward_preds[0, 2])
        else:
            if random.random() > epsilon:
                real_act += 1
                reward_preds = agent.get_qs(current_state)
                best_action = np.argmax(reward_preds)
            else:
                best_action = random.choices([0, 1, 2])[0]
 
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
        actions += 1
        tot_reward += reward
 
        #print(f"{episode} Day {actions} {best_action}: {dct['active_cases']}, {dct['economy']}, {reward}")
        agent.update_replay_memory((current_state, best_action, reward, new_state, done))
        agent.train(done, actions)
        current_state = new_state
    
    # Just an optimization
    if epi%10 != 0: agent.train_chain()
    
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= epi >= START_EPSILON_DECAYING and epsilon > 0.2:
        epsilon -= epsilon_decay_value
    
    if epi%10 == 0:
        saveFile(epi, DEF_SDIR, saveIter=10)
    
    rl = len(data['Reward'])
    data['RewardSum'] = [x for x in data['Reward']]
    #data['RewardSum'] = list(agent.norm(np.array(data['RewardSum'])))
    for i in range(rl-2, -1, -1):
        data['RewardSum'][i] += data['Reward'][i+1]*agent.DISCOUNT
    data['Total Infected'] = env.total_infected
    data['Total Cured'] = env.total_cured
    data['Total Death'] = env.total_death
    data['Total Economy'] = np.sum(data['Economy'])
    data['MAX_ECONOMY'] = env.observation_space_high[2]
    data['TOTAL_POPULATION'] = env.observation_space_high[1]
    rewardsum = np.sum(data['Reward'])
    
    if epi%10 == 0:
        agentAVG.append(rewardsum)
    avgreward = np.sum(agentAVG)/max(len(agentAVG), 1)

    if epi%5 == 0:
        if epi%10 == 0: 
            draw(data, rewards=True, plot=False, 
                title=f"Episode {epi}, Actions:{real_act}-{actions} Time:{time.time()-st:0.1f}s, Reward:{rewardsum:0.1f}, AVG:{avgreward:0.1f}",
                filename=os.path.join(os.path.realpath('.'), 'train_logs', f'train_{epi}_full_agent'))
        else:
            draw(data, rewards=False, plot=False, 
                title=f"Episode {epi}, Actions:{real_act}-{actions} Time:{time.time()-st:0.1f}s, Reward:{rewardsum:0.1f}, AVG:{avgreward:0.1f}",
                filename=os.path.join(os.path.realpath('.'), 'train_logs', f'train_{epi}'))
    
    if epi%10 != 0: randagentAVG.append(rewardsum)
    
    avgrandreward = np.sum(randagentAVG)/len(randagentAVG)
    print(f"\rEpisode:{epi}, Actions:{real_act}-{actions}, Time:{time.time()-st:0.1f}s, Reward:{rewardsum:0.1f}, AVG:{avgreward:0.1f}, rAVG:{avgrandreward:0.1f}", end='')
