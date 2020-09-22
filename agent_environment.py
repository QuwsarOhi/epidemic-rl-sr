import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
#%matplotlib inline

def draw(data, filename=None, title='', rewards=False, plot=True):
    sns.set_style("whitegrid")
    handles = []
    handels_label = []
    def win(v):
        pst = v[0]
        pos = 0
        done = False
        c = ['#17EB53', 'blue', '#FF0080']
        dc = [0, 0, 0]
        label = ['Level-0 Restriction', 'Level-1 Restriction', 'Level-2 Restriction']
        for i in range(1, len(v)):
            if v[i] != pst:
                plt.axvspan(pos+1, i+1, alpha=0.2, color=c[pst])
                if label[pst] not in handels_label: 
                    handles.append(mpatches.Patch(color=c[pst], label=label[pst], alpha=0.2))
                    handels_label.append(label[pst])
                dc[pst] += 1
                pos, pst = i, v[i]
        plt.axvspan(pos+1, len(v), alpha=0.2, color=c[pst], 
                    label = '_'*dc[pst] + label[pst])
        if label[pst] not in handels_label:
            handles.append(mpatches.Patch(color=c[pst], label=label[pst], alpha=0.2))
            handels_label.append(label[pst])

    clear_output(wait=True)
    ticks = np.arange(0, max(data['Day'])+1, 40)
    w, h = (14, 7)
    plt.figure(figsize=(w, h), dpi=400,)
    pop = data['Population']
    TiP = data['Total Infected']/pop * 100
    TdP = data['Total Death']/pop * 100
    info = f"Total Infected: {data['Total Infected']} ({TiP:.1f}%)" + \
           f", Total Death: {data['Total Death']} ({TdP:.1f}%)" #+ \
           #f", Total Economy: {int(data['Total Economy'])}"
    if title != '': title += '\n' + info
    #else:           title = info
    #print(title)
    plt.suptitle(title, y=0.95)

    # Subplot 1
    plt.subplot(2, 2, 1)
    win(data['Action'])
    if rewards:
        plt.plot(data['Day'], data['Reward'], label='Reward', c='g')
        handles.append(mlines.Line2D([], [], color='g', label='Reward'))
        if 'RewardSum' in data:
            plt.plot(data['Day'], data['RewardSum'], '--', c='g', label='Chain Reward')
            handles.append(mlines.Line2D([], [], color='g', label='Chain Reward'))
        if 'Action1' in data:
            plt.plot(data['Day'], data['Action1'], '--', label='Action1 Reward', c='c')
            handles.append(mlines.Line2D([], [], linestyle='--', color='c', label='Action1 Reward'))
        if 'Action2' in data:
            plt.plot(data['Day'], data['Action2'], '--', label='Action2 Reward', c='m')
            handles.append(mlines.Line2D([], [], linestyle='--', color='m', label='Action2 Reward'))
        if 'Action3' in data:
            plt.plot(data['Day'], data['Action3'], '--', label='Action3 Reward', c='y')
            handles.append(mlines.Line2D([], [], linestyle='--', color='y', label='Action3 Reward'))
        plt.ylabel('Point')
    else:
        plt.plot(data['Day'], (np.cumsum(data['Infected'])/pop)*100, 
                 label='Infected (Cumulative)', c='#C76B01')
        handles.append(mlines.Line2D([], [], color='#C76B01', label='Infected (Cumulative)'))
        plt.plot(data['Day'], (np.cumsum(data['Cured'])/pop)*100,
                 label='Cured (Cumulative)', c='#0F9E0D')
        handles.append(mlines.Line2D([], [], color='#0F9E0D', label='Cured (Cumulative)'))
        plt.plot(data['Day'], (np.cumsum(data['Death'])/pop)*100,
                 label='Death (Cumulative)', c='#C2000F')
        handles.append(mlines.Line2D([], [], color='#C2000F', label='Death (Cumulative)'))
        plt.ylabel('Population (%)')
    
    #plt.axes.get_yaxis().set_visible(False)
    #plt.xlabel('Days')
    #plt.xticks(ticks)
    plt.grid(alpha=0.8, which='major')
    plt.grid(alpha=0.8, which='minor', linestyle=':')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    #plt.legend(prop={'size': 8})

    # Subplot 2
    plt.subplot(2, 2, 2)
    plt.plot(data['Day'], data['R0'], label='Reproduction Rate ($R_0$)', c='#B032C2')
    #plt.plot(data['Day'], data['R00'], '--', label='Total Reproduction Rate', c='#B032C2')
    handles.append(mlines.Line2D([], [], color='#B032C2', label='Reproduction Rate ($R_0$)'))
    #handles.append(mlines.Line2D([], [], linestyle='--', color='#B032C2', label='Total Reproduction Rate'))

    win(data['Action'])
    plt.axhline(1.0, linestyle=':', lw=1)
    #plt.xlabel('Days')
    plt.ylabel('$R_0$')
    plt.grid(alpha=0.8, which='major')
    plt.grid(alpha=0.8, which='minor', linestyle=':')
    #plt.legend(prop={'size': 8})
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    # Subplot 3
    plt.subplot(2, 2, 3)
    plt.plot(data['Day'], (np.array(data['Active Cases'])/pop)*100, label='Active Cases', c='#2E4CC2')
    handles.append(mlines.Line2D([], [], color='#2E4CC2', label='Active Cases'))
    handles.append(mlines.Line2D([], [], color='#089E87', label='Economy'))
    plt.xlabel('Days')
    plt.ylabel('Population (%)')
    win(data['Action'])
    #plt.xticks(ticks)
    plt.grid(alpha=0.8, which='major')
    plt.grid(alpha=0.8, which='minor', linestyle=':')
    #plt.legend(prop={'size': 8})
    plt.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.10, -0.42),
               ncol=5)
    # Subplot 4
    plt.subplot(2, 2, 4)
    tmpEco = [data['MAX_ECONOMY'] for i in range(6)] + data['Economy']
    window = np.convolve(tmpEco/data['MAX_ECONOMY'] , 
                         np.ones(7, dtype=int), 'valid')/7
    plt.plot(data['Day'], window, label='Economy', c='#089E87')
    plt.ylim((0, 1))
    plt.xlabel('Days')
    plt.ylabel('Economy Ratio')
    
    win(data['Action'])
    #plt.xticks(ticks)
    plt.grid(alpha=0.8, which='major')
    plt.grid(alpha=0.8, which='minor', linestyle=':')
    #plt.legend(prop={'size': 8})
    

    plt.subplots_adjust(wspace=w*0.009, hspace=h*0.005)
    if filename: plt.savefig(f'{filename}.pdf', dpi=400, bbox_inches='tight', transparent=True)
    if plot: plt.show()
    else: plt.close('all')


import numpy as np
import os
import pickle
import time
from math import ceil
import random
from glob import glob
from random import Random
myRandom = Random()
 
class Person():
    cure_after = 0
    infectionSpread = 0
    death_distribution = []
 
    def __init__(self, beta, ):
        self.alive = True
        self.isCured = False
        self.infectionDay = 0
        self.pos = None
        self.beta = beta
 
    
    def daily_process(self):
        if self.infectionDay >= 1:
            self.infectionDay += 1
        if self.infectionDay >= self.cure_after:
            if myRandom.choices([True, False], weights=[0.4, 0.6]) == False:
                return None
            should_die = myRandom.choices(population=[True, False], 
                                        weights=self.death_distribution,
                                        k=1)[0]
            if should_die:
                self.alive = False
                self.infectionDay = 0
                self.isCured = True
                return 'dead'
            else:
                self.infectionDay = 0
                self.isCured = True
                return 'cured'
        return None
 
    def makeInfected(self):
        self.infectionDay = 1
 
    def makeInstantInfected(self):
        self.infectionDay = self.infectionSpread
    
    def isInfected(self):
        return self.infectionDay >= 1
 
    def canInfect(self):
        return self.infectionDay >= self.infectionSpread
    
    def movement_economy(self, moves, multiplier=1):
        return moves*multiplier*myRandom.uniform(0.7, 0.9)


class Env:
    def __init__(self, *args, **kwargs):
        '''
        height = height of the map
        width = width of the map
        population_density = density of the population, 
                             where density = population/area
        '''
        self.kwargs         = kwargs
        self.height         = int(kwargs['height'])
        self.width          = int(kwargs['width'])
        self.area           = int(self.height*self.width)
        if 'population_density' in kwargs:
            self.pd             = kwargs['population_density']
        if 'population' in kwargs:
            self.population = kwargs['population']
            self.pd = self.area/self.population
        else:
            self.population     = int(self.pd*self.area)
        self.first_infected = int(ceil(self.population*kwargs['infected_ratio']))
        self.graph          = None
        self.person         = None
        self.state_log      = None
        self.boundary       = int(kwargs['boundary'])
        self.beta           = kwargs['beta'] if 'beta' in kwargs else np.inf
        self.prob           = kwargs['prob']
        self.day            = None
        self.day_step       = kwargs['day_step']
        self.days           = kwargs['day_limit']
        self.infected_id    = None
        self.infection_graph = None
        self.total_infected = None
        self.total_cured    = None
        self.total_death    = None
        self.population_multiplier = [[4, 1], [2, 0.75], [1, 0.65]]
        self.q              = -0.7
        self.r              = -8
        self.linear_reward  = False
        self.state_disconnect = False
        self.observation_space_high = np.array([self.days, 
                                                self.population, 
                                                self.day_step*self.population*0.9], 
                                                dtype=np.float32)
        self.observation_space_low = np.array([0, 0, 0.25*self.population/2], 
                                              dtype=np.float32)
 
        if 'death_distribution' in kwargs:
            Person.death_distribution = kwargs['death_distribution']
        if 'cure_after' in kwargs:
            Person.cure_after = kwargs['cure_after']
        if 'infect_after' in kwargs:
            Person.infectionSpread = kwargs['infect_after']
 
        self.obj            = {'empty' : -1}
        self.moves          = [[-1, -1], [-1, 0], [-1, 1],
                               [0, -1], [0, 0], [0, 1],
                               [1, -1], [1, 0], [1, 1]]
 
    def _place_person(self, high_x, high_y, size, low_x=0, low_y=0):
        '''Generates persons in random places'''
        person_pos = np.zeros((size, 2))
        pos_found = 0
        while pos_found < size:
            x = myRandom.randint(low_x, high_x-1)
            y = myRandom.randint(low_y, high_y-1)
 
            if self.graph[x][y] == self.obj['empty']:
                person_pos[pos_found, 0] = x
                person_pos[pos_found, 1] = y
                pos_found = pos_found + 1
        return person_pos
 
 
    def _gen_population(self):
        '''the main population generation method'''
        self.population -= self.population%5
        sz = self.population//5
        person_pos = np.zeros((sz*5, 2), dtype=np.int16)
        
        person_pos[:, :] = self._place_person(high_x = self.height,
                                              high_y = self.width,
                                              size = self.population)
            
        # Place the population in graph
        self.person = [Person(self.beta) for i in range(self.population)]
        for it in range(self.population):
            x = person_pos[it][0]
            y = person_pos[it][1]
            self.graph[x][y] = it
            self.person[it].pos = [x, y] 
 
 
    def gen_image(self, returnImg=False):
        imgShape = (self.height, self.width, 3)
        image = np.zeros(imgShape, dtype=np.float)
 
        for x in range(self.height):
            for y in range(self.width):
                if self.graph[x][y] == self.obj['empty']:
                    continue
                else:
                    it = self.graph[x][y]
                    if self.person[it].isInfected():
                        image[x, y, :] = to_rgb('r')
                    elif self.person[it].isCured:
                        image[x, y, :] = to_rgb('lime')
                    else:
                        image[x, y, :] = to_rgb('y')
        if returnImg == False:
            plt.imshow(image, interpolation='nearest', aspect='auto')
            plt.axis('off')
        else:
            return image
 
 
    '''-------------- core methods for population movement -----------------'''
    # Checks if a position is valid
    def _is_valid(self, pos):
        if (0 <= pos[0] < self.height) and (0 <= pos[1] < self.width):
            if self.state_disconnect and ((pos[0] == self.height_half) or (pos[1] == self.width_half)):
                return False
            return True
        return False
 
 
    def _move_person(self, it):
        if (self.person[it].alive == False):
            return
        while True:
            d = myRandom.randint(0, len(self.moves)-1)
            if d == 4: break
            new_pos = list(map(sum, zip(self.person[it].pos, self.moves[d])))
            if not self._is_valid(new_pos): continue
            if (self.graph[new_pos[0]][new_pos[1]] >= 0) and (self.person[it].isInfected() == True):
                self._try_infect(it, self.graph[new_pos[0]][new_pos[1]])
                continue
 
            self.graph[self.person[it].pos[0]][self.person[it].pos[1]] = self.obj['empty']
            self.person[it].pos = new_pos
            self.graph[new_pos[0]][new_pos[1]] = it
            return
 
 
    def _move_population(self, lockdown_state=1.0):
        '''Moves the total population by n random steps'''
        steps = int(lockdown_state*self.day_step)
        for step in range(steps):
            for it in range(self.population):
                self._move_person(it)
        
        # Economy calculation
        for it in range(self.population):
            if self.person[it].canInfect() or (self.person[it].alive == False):
                continue
            self.log['economy'] += self.person[it].movement_economy(moves=steps)
 
 
    def _iter_pos(self, pos):
        '''An iterator that returns all valid and possible moves'''
        for it in range(len(self.moves)):
            dx = pos[0] + self.moves[it][0]
            dy = pos[1] + self.moves[it][1]
            if self._is_valid((dx, dy)):
                yield it, dx, dy
    
 
    def _check_surrounding(self, it0):
        '''each person checks his surrounding to spread infection,
        if anyone is not infected around this person, he/she may be infected'''
        for dir_id, x, y in self._iter_pos(self.person[it0].pos):
            if (self.graph[x][y] >= 0) and (dir_id != 4):
                it1 = self.graph[x][y]
                if (not self.person[it1].isInfected()) and (not self.person[it1].isCured):
                    state = myRandom.choices(population=[True, False], 
                                           weights=[self.prob, 1-self.prob],
                                           k=1)[0]
                    if state == True:
                        self.person[it1].makeInfected()
                        self.log['infected'] += 1
                        self.infected_id.add(it1)
                        self.infection_graph[it0].append(it1)
 
                        self.person[it0].beta -= 1
                        if self.person[it0].beta > 0:
                            return
    
    def _try_infect(self, it0, it1):
        if self.person[it0].beta <= 0:
            return
        if (not self.person[it1].isInfected()) and (not self.person[it1].isCured):
            state = myRandom.choices(population=[True, False], 
                                   weights=[self.prob, 1-self.prob], k=1)[0]
            if state == True:
                self.person[it1].makeInfected()
                self.log['infected'] += 1
                self.infected_id.add(it1)
                self.infection_graph[it0].append(it1)
                self.person[it0].beta -= 1
                
 
    '''---------------- driver methods of ENV -------------------------------'''
    def reset_env(self, seed=None, verbose=False):
        if seed is not None:
            myRandom.seed(seed)
 
        self.day = 1
        self.total_infected = 0
        self.total_cured = 0
        self.total_death = 0
        self.graph = [[self.obj['empty']]*self.width for i in range(self.height)]
        self.infected_id = set()
        self._gen_population()
        self.infection_graph = [[] for i in range(self.population)]
        self._infect_random()
        
        if verbose:
            print('Total Area', self.area)
            print('Total Population', self.population)
            print('First Infected', len(self.infected_id))
        return
 
 
    def runDay(self, lockdown_state=1, state_disconnect=False):
        self.log = {"day":self.day, "infected":0, "death":0, "cured":0, 
                    "active_cases":0, "economy":0}
 
        self.state_disconnect = state_disconnect
        self._move_population(lockdown_state)
 
        # Health Check
        temps = []
        for it in self.infected_id:
            ret = self.person[it].daily_process()
            if ret == None:
                continue
            temps.append(it)
            if ret == 'dead':
                x, y = self.person[it].pos
                self.graph[x][y] = self.obj['empty']
                self.log['death'] += 1
            else:
                self.log['cured'] += 1
        
        # Removing the cured persons
        for it in temps:
            self.infected_id.remove(it)
        self.log['active_cases'] = len(self.infected_id)
        self.total_infected += self.log['infected']
 
        # Calculating Delta Reproduction rate
        R0, R00 = 0, 0
        for it in range(self.population):
            if len(self.infection_graph[it]) > 0 and it in self.infected_id:
                R0 += 1
            if len(self.infection_graph[it]) > 0:
                R00 += 1
        
        if R00 != 0:
            R00 = self.total_infected/R00

        if R0 != 0:
            R0 = len(self.infected_id)/R0

        self.day += 1
        self.log['R0'] = R0
        self.log['R00'] = R00
        self.total_cured += self.log['cured']
        self.total_death += self.log['death']
        return self.log
 
 
    def reward(self, action):
        ac = self.log['active_cases']
        eco = self.log['economy']
        dr = self.log['death']/self.population
        td, tp, te = self.observation_space_high
        ac = (ac/tp)*100
        eco = (eco/te)
        lmin = -10
        lmax = 1
        df = 5
        if self.kwargs.get('linear_reward'):
            ret = (self.q*self.population_multiplier[action][0]*ac + self.population_multiplier[action][1])*eco
            if self.kwargs.get('normalize'): return (ret - lmin) / (lmax - lmin)
            else: return ret
        else:
            return np.exp(self.r*ac)*eco - dr*df
        
 
    def get_state(self):
        td, tp, te = self.observation_space_high
        new_state = np.array([self.log['active_cases']/tp,
                              self.log['infected']/tp,
                              self.total_cured/tp,
                              self.total_death/tp,
                              self.log['economy']/te,
                              self.log['R0']], 
                             dtype=np.float32)
        return new_state
 
    def step(self, action, staetlock=False):
        lockdown_state = [1.0, 0.75, 0.25]
        self.state_disconnect = staetlock
        ret = self.runDay(lockdown_state[action])
        reward = self.reward(action)
        done = True if ret['active_cases'] == 0 else False
        done = done and (self.log['day'] >= self.days)
        self.state_disconnect = False
        #if done: reward = 1
        return (self.get_state(), reward, done, ret)
 
    
    def reset(self, seed=None, verbose=False):
        self.reset_env(seed, verbose)
        self.runDay(1)
        return np.array(self.get_state(), dtype=np.float32)
 
 
    def _infect_random(self, ):
        infected_population = myRandom.sample(range(0, self.population), 
                                            self.first_infected)
        for it in infected_population:
            self.person[it].makeInstantInfected()
            self.infected_id.add(it)


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
        agent.model.save_weights(sdir+f'agent_{epi}.h5')
    
    agent.model.save_weights(sdir+'model.h5')
    agent.target_model.save_weights(sdir+'tmodel.h5')
    with open(sdir+f'replay.pkl', 'wb') as f:
        pickle.dump(agent.replay_memory, f)
 
def loadFile(sdir, loadParam=True):
    files = glob(sdir+'*.pkl', recursive=False)
    if len(files) == 0:
        print('No files found...')
    else:
        print('Files found:', [str(a).split('/')[-1] for a in files])
 
    global agent
    global epsilon
    global START_EPSILON_DECAYING
    global END_EPSILON_DECAYING
    global epsilon_decay_value
    global episode
 
    agent.model.load_weights(sdir+'model.h5')
    agent.target_model.load_weights(sdir+'tmodel.h5')
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


from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import random
from math import ceil
 
class DQN:
    def __init__(self, input_shape, action_size, timestep, 
                 model_path=None, **kwargs):
        # Minimum number of steps in a memory to start training
        if 'REPLAY_MEM_SZ' in kwargs:
            self.REPLAY_MEM_SZ = kwargs['REPLAY_MEM_SZ']
        else:
            self.REPLAY_MEM_SZ = 1000
        # Batch size of training
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 32
        # The number of episodes after which, the target model is updated
        if 'UPDATE_TARGET' in kwargs:
            self.UPDATE_TARGET = kwargs['UPDATE_TARGET']
        else:
            self.UPDATE_TARGET = 5
        # Setting DISCOUNT value that fetches future rewards
        if 'DISCOUNT' in kwargs:
            self.DISCOUNT = kwargs['DISCOUNT']
        else:
            self.DISCOUNT = 0.95
        if 'DISCOUNT_DECAY' in kwargs:
            self.DISCOUNT_DECAY = kwargs['DISCOUNT_DECAY']
        else:
            self.DISCOUNT_DECAY = 0.05
        self.input_shape = input_shape
        self.timestep = timestep
 
        # Main model
        if model_path != None:
            model = load_model(model_path)
        self.model = self.create_model(input_shape, action_size)
    
        # Target model
        self.target_model = self.create_model(input_shape, action_size)
        self.target_model.set_weights(self.model.get_weights())
 
        # An array containing last n steps for training
        self.replay_memory = deque(maxlen=2*self.REPLAY_MEM_SZ)
 
        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
 
        self.initReplay()

    def initReplay(self):
        self.replay_memory.clear()
        for i in range(self.REPLAY_MEM_SZ):
            self.replay_memory.append((np.ones(self.input_shape[-1]),
                                       2, 0.5, np.ones(self.input_shape[-1]),
                                       True))   
 
    # Input shape = (TimeStep, Features) : (None, Features)
    def create_model(self, input_shape, action_size):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, 
                       bias_initializer='he_uniform', kernel_initializer='he_uniform',),
                       input_shape=input_shape))
        model.add(Bidirectional(LSTM(64, dropout=0.1, bias_initializer='he_uniform',
                       return_sequences=True, kernel_initializer='he_uniform')))
        model.add(Bidirectional(LSTM(64, dropout=0.1, bias_initializer='he_uniform',
                       return_sequences=False, kernel_initializer='he_uniform')))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform',
                        bias_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform',
                        bias_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform',
                        bias_initializer='he_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(action_size, activation='linear', 
                        kernel_initializer='he_uniform', 
                        bias_initializer='he_uniform'))
        lr_schedule = ExponentialDecay(initial_learning_rate=0.1,
                                       decay_steps=1500, 
                                       decay_rate=0.8)
        adam = Adam(learning_rate=0.001)#lr_schedule)
        model.compile(loss="mse", optimizer=adam)
        return model
 
    # Adds step's data to a memory replay array
    # (current_state, action, reward, new_state, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def norm(self, x):
        return ((x[:] - x.min()) / (x.max() - x.min()))
 
    # Fetches states including previous timesteps
    # State: ACr, INFr, TCr, TDr, ECOr, R0, ACT 
    def fetch(self, index, trans, timestep=None):
        if timestep == None: timestep = self.timestep
        st, ed = max(index-timestep+1, 0), index
        ret = [self.replay_memory[i][trans] for i in range(st, ed)]
        done = False
        p, q = len(ret)-1, 0
        for i in range(ed, st+1, -1):
            done = (self.replay_memory[i][-1]) or done
            if done: ret[p] = ret[q]
            if not done: q = p
            p -= 1
        return ret

    # Not the original train function, but accelerates the output process
    def train(self, terminal_state, step):
        if len(self.replay_memory) < 2*self.REPLAY_MEM_SZ:
            return
        
        # Fetching a batch of randomly selection history
        minibatch_index = random.sample(range(self.REPLAY_MEM_SZ,
                                              2*self.REPLAY_MEM_SZ),
                                              self.batch_size)
        # The current_state data
        current_states = np.array([self.fetch(i, 0) for i in minibatch_index], 
                                  dtype=np.float32)
        current_qs_list = self.model.predict(current_states)
 
        # The future_state data
        future_states = np.array([self.fetch(i, 3) for i in minibatch_index], 
                                  dtype=np.float32)
        future_qs_list = self.target_model.predict(future_states)
 
        updated_reward = np.zeros(current_qs_list.shape)
        for pos, index in enumerate(minibatch_index):
            (current_state, action, reward, future_state, done) = self.replay_memory[index]
            if not done:
                new_q = reward + self.DISCOUNT * np.max(future_qs_list[pos])
            else:
                new_q = reward
 
            updated_reward[pos] = current_qs_list[pos]
            updated_reward[pos, action] = new_q
 
        self.model.fit(current_states, updated_reward, batch_size=16, verbose=0)

        if terminal_state:
            self.target_update_counter += 1
 
        if self.target_update_counter >= self.UPDATE_TARGET:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
   
    
    def train_chain(self):
        if len(self.replay_memory) < self.REPLAY_MEM_SZ:
            return
        pos = len(self.replay_memory)-1
        chain_reward = [self.replay_memory[pos][2]]
        states = [self.fetch(pos, 0)]
        actions = [self.replay_memory[pos][1]]
        pos -= 1
 
        while not self.replay_memory[pos][-1]:
            reward = self.replay_memory[pos][2] + self.DISCOUNT * chain_reward[-1]
            chain_reward.append(reward)
            states.append(self.fetch(pos, 0))
            actions.append(self.replay_memory[pos][1])
            pos -= 1
 
        chain_reward = np.array(chain_reward)
        states = np.array(states)
        actions = np.array(actions)
        pred_rewards = self.target_model.predict(states)
        for i, (action, reward) in enumerate(zip(actions, chain_reward)):
            pred_rewards[i, action] = reward
 
        #pred_rewards = self.norm(pred_rewards)
        self.target_model.fit(states, pred_rewards, batch_size=16, epochs=4, 
                              verbose=0)
        self.model.fit(states, pred_rewards, batch_size=16, epochs=4, 
                              verbose=0)    
    
    def get_qs(self, state):
        pos = len(self.replay_memory)-1
        ret = self.fetch(pos, 0)
        ret.pop(0)
        ret.append(state)
        ret = np.array(ret, dtype=np.float)
        ret = ret[np.newaxis, :]
        return self.model.predict(ret)