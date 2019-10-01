import numpy as np
import time
from pathlib import Path
import pandas as pd
import os
import copy
import pyphi
import networkx as nx

from actual_agency import *

class Animat:
    '''
    This class contains functions concerning the animat to be analyzed
    '''

    def __init__(self, params):
        '''
        Function for initializing the animat.
        Called by pyanimats.Animat(params)
            Inputs:
                params: a dictionary containing the defining parameters of the animat. The minimal input is {}
            Outputs:
                updates the animat object (self) directly
        '''

        # checking if params contains the right keys, else using standard values
        self.n_left_sensors = params['nrOfLeftSensors'] if 'nrOfLeftSensors' in params else 1
        self.n_right_sensors = params['nrOfRightSensors'] if 'nrOfRightSensors' in params else 1
        self.n_hidden = params['hiddenNodes'] if 'hiddenNodes' in params else 4
        self.n_motors = params['motorNodes'] if 'motorNodes' in params else 2
        self.gapwidth = params['gapWidth'] if 'gapWidth' in params else 1
        self.n_sensors = self.n_right_sensors + self.n_left_sensors
        self.n_nodes = self.n_sensors + self.n_hidden + self.n_motors
        self.length = self.n_left_sensors  + self.gapwidth + self.n_right_sensors
        self.x = params['x'] if 'x' in params else 0
        self.y = params['y'] if 'y' in params else 0

    def __len__(self):
        return self.length

    def set_x(self, position):
        # function for setting the current x position of the animat
        self.x = position

    def set_y(self, position):
        # function for setting the current y position of the animat
        self.y = position


    def _getBrainActivity(self,data):
        '''
        Function for initializing the animat.
        Called by pyanimats.Animat(params)
            Inputs:
                data: the unpickled output from MABEs markov_io_map from markov gates
            Outputs:
                brain_activity: 3D np.array with binary activity of the nodes (trials x times x nodes)
        '''

        world_height = 34
        print('Creating activity matrix from MABE otput...')
        n_trials = int((np.size(data,0))/world_height)
        brain_activity = np.zeros((n_trials,1+world_height,self.n_nodes))

        for i in list(range(n_trials)):
            for j in list(range(world_height+1)):
                ix = i*world_height + j
                # reading out node activities from MABE output.
                # and shifting the time of hidden and motor activation to reflect that they are affected by sensors
                if j==0:
                    # Speial case for first timestep
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:self.n_sensors]
                    hidden = np.zeros(self.n_hidden)
                    motor = np.zeros(self.n_motors)
                elif j==world_height:
                    # special case for last timestep
                    sensor = np.zeros(self.n_sensors)
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                else:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:self.n_sensors]
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                nodes = np.r_[sensor, motor, hidden]
                brain_activity[i,j,:] = nodes
        print('Done.')

        # return output
        return brain_activity


    def get_state(self, trial, t):
        '''
        Function for picking out a specific state of the system.
         Inputs:
             trial: the trial number that is under investigation (int)
             t: the timestep you wish to find the transition to (int)
         Outputs:
             two tuples (X and Y in Albantakis et al 2019) containing the state of the system at time t-1 and t.
        '''
        # Checking if brain activity exists
        if not hasattr(self, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')

        # return state as a tuple (can be used for calculating Phi)
        return tuple(self.brain_activity[trial, t].astype(int))



    def get_unique_states(self, trial= None):
        '''
        Function for getting all unique transitions a system goes through in its lifetime.
            Inputs:
                trial: the number of a specific trial to investigate (int, if None then all trials are considered)
            Outputs:
                unique_states: a list of all unique states found
        '''
        if not hasattr(self, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')

        n_trials = self.brain_activity.shape[0]
        n_times = self.brain_activity.shape[1]

        trials = range(n_trials) if trial==None else [trial]
        unique_states = []
        for trial in trials:
            for t in range(0,n_times):
                state = self.get_state(trial, t)
                if state not in unique_states:
                    unique_states.append(state)

        return unique_states


    def enumerate_states(self):
        '''
        Function for enumerating states, to avoid calculating causal accounts too many times
            Inputs:

            Outputs:
                no output, just an update of the Animat object
        '''
        # Check if brain activity exists
        if not hasattr(self, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')

        n_trials = self.brain_activity.shape[0]
        n_times = self.brain_activity.shape[1]
        enumerated_states = []

        # looping through trials and time points
        for trial in range(n_trials):
            for t in range(n_times):
                # getting current transition and checking if it is new
                state = self.get_state(trial, t)
                n = state2num(list(state))
                # adding transition idendtifier to list
                enumerated_states.append(n)

        self.enumerated_states = enumerated_states


    def get_transition(self, trial, t, trim=False):
        '''
        Function for picking out a specific transition: state(t-1) --> state(t).
            Inputs:
                trial: the trial number that is under investigation (int)
                t: the timestep you wish to find the transition to (int)
                trim: True if the transition should not contain motors in t-1 or sensors in t.
                ### IF YOU USE TRIM, MAKE SURE TO CHECK THE CODE BELOW ###
            Outputs:
                two tuples (X and Y in Albantakis et al 2019) containing the state of the system at time t-1 and t.
        '''

        # Exceptions
        if not hasattr(self, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')
        if t > self.brain_activity.shape[1]:
            raise IndexError('t is beyond number of times in brain activity, no transition here.')
        if t == 0:
            raise IndexError('t==0, no transition here.')

        # Returning outputs depending on the Trim option (THIS MUST BE CHECKED)
        if trim:
            sensor_ixs = list(range(self.n_sensors))
            motor_ixs = list(range(self.n_sensors,self.n_sensors+self.n_motors))
            hidden_ixs = list(range(self.n_sensors+self.n_motors,self.n_sensors+self.n_motors+self.n_hidden))
            before_state_ixs = sensor_ixs.extend(hidden_ixs)
            after_state_ixs  = motor_ixs.extend(hidden_ixs)
            #before_state_ixs = [0,1,4,5,6,7] if self.n_nodes==8 else [0,3,4,5,6] # old specialized code
            #after_state_ixs  = [2,3,4,5,6,7] if self.n_nodes==8 else [1,2,3,4,5,6]
            return tuple(self.brain_activity[trial, t-1, before_state_ixs].astype(int)), tuple(self.brain_activity[trial, t, after_state_ixs].astype(int))
        else:
            return tuple(self.brain_activity[trial, t-1].astype(int)), tuple(self.brain_activity[trial, t].astype(int))


    def get_unique_transitions(self, trial=None, trim=True):
        '''
        Function for getting all unique transitions a system goes through in its lifetime.
            Inputs:
                trial: the number of a specific trial to investigate (int, if None then all trials are considered)
                trim: True if the transition should not contain motors in t-1 or sensors in t.
            Outputs:
                unique_transitions: a list of all unique transitions found
                unique_idxs: list of indices of the unique transitions' first occurence
        '''

        # Check if brain activity exists
        if not hasattr(self, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')

        # setting required variables and output lists
        n_trials = self.brain_activity.shape[0]
        n_times = self.brain_activity.shape[1]
        unique_transitions = []
        unique_transitions_compressed = []
        unique_idxs = []

        # defining the trials that will be searched
        trials = range(n_trials) if trial==None else [trial]

        # looping through trials and time points
        for trial in trials:
            for t in range(1,n_times):
                # getting current transition and checking if it is new
                transition = self.get_transition(trial, t, trim)
                if transition not in unique_transitions:
                    unique_transitions.append(transition)
                    unique_idxs.append((trial, t))

        # picking unique transitions without trimming is trim = False
        if not trim:
            unique_transitions = [self.get_transition(trial, t, trim=False) for trial,t in unique_idxs]

        # reutrn outputs
        return unique_transitions, unique_idxs

    def enumerate_transitions(self):
        '''
        Function for enumerating transitions, to avoid calculating causal accounts too many times
            Inputs:

            Outputs:
                no output, just an update of the Animat object
        '''

        # Check if brain activity exists
        if not hasattr(self, 'brain_activity'):
            raise AttributeError('No brain activity saved yet.')

        n_trials = self.brain_activity.shape[0]
        n_times = self.brain_activity.shape[1]
        enumerated_transitions = []

        # looping through trials and time points
        for trial in range(n_trials):
            for t in range(1,n_times):
                # getting current transition and checking if it is new
                transition = self.get_transition(trial, t, trim=False)
                # adding transition idendtifier to list
                enumerated_transitions.append(state2num(list(transition[0])+list(transition[1])))

        self.enumerated_transitions = enumerated_transitions


    def saveBrainActivity(self, brain_activity):
        '''
        More general function for saving brain activity to animat object
            Inputs:
                trial: brain activity, either as MABE output or 3D array (trials x times x nodes)
            Outputs:
                no output, just an update of the Animat object
        '''
        # call getBrainActivity function if brain activity is pandas
        if type(brain_activity)==pd.core.frame.DataFrame:
            self.brain_activity = self._getBrainActivity(brain_activity)
        else: ## if brain activity is array form
            assert brain_activity.shape[2]==self.n_nodes, "Brain history does not match number of nodes = {}".format(self.n_nodes)
            self.brain_activity = np.array(brain_activity)
            self.n_trials = brain_activity.shape[0]
            self.n_timesteps = brain_activity.shape[1]

    def saveUniqueStates(self):
        unique_states = self.get_unique_states()
        self.unique_states = unique_states
        self.enumerate_states()
        self.num_unique_states = len(unique_states)

    def saveUniqueTransitions(self):
        self.unique_transitions, self.unique_idxs = self.get_unique_transitions(trim=False)
        self.enumerate_transitions()
        self.num_unique_transitions = len(self.unique_transitions)

    def save_unique_causal_links(self):
        self.causal_links = get_all_causal_links(self)


    def saveBrain(self, TPM, cm, node_labels=[]):
        '''
        Function for giving the animat a brain (pyphi network) and a graph object
            Inputs:
                TPM: a transition probability matrix readable for pyPhi
                cm: a connectivity matrix readable for pyPhi
                node_labels: list of labels for nodes (if empty, standard labels are used)
            Outputs:
                no output, just an update of the animat object
        '''
        if not len(node_labels)==self.n_nodes:
            node_labels = []
            # standard labels for up to 10 nodes of each kind
            sensor_labels = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
            motor_labels = ['M1','M2','M3','M4','M5','M6','M7','M8','M9','M10']
            hidden_labels = ['A','B','C','D','E','F','G','H','I','J']

            # defining labels for each node type
            s = [sensor_labels[i] for i in list(range(self.n_sensors))]
            m = [motor_labels[i] for i in list(range(self.n_motors))]
            h = [hidden_labels[i] for i in list(range(self.n_hidden))]

            # combining the labels
            node_labels.extend(s)
            node_labels.extend(m)
            node_labels.extend(h)

        # defining the network using pyphi
        network = pyphi.Network(TPM, cm, node_labels=node_labels)
        self.brain = network
        self.TPM = TPM
        self.cm = cm

        # defining a graph object based on the connectivity using networkx
        G = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())
        mapping = {key:x for key,x in zip(range(self.n_nodes),node_labels)}
        G = nx.relabel_nodes(G, mapping)
        self.brain_graph = G

        # saving the labels and indices of sensors, motors, and hidden to animats
        self.node_labels = node_labels
        self.sensor_ixs = list(range(self.n_sensors))
        self.sensor_labels = [node_labels[i] for i in self.sensor_ixs]
        self.motor_ixs = list(range(self.n_sensors,self.n_sensors+self.n_motors))
        self.motor_labels = [node_labels[i] for i in self.motor_ixs]
        self.hidden_ixs = list(range(self.n_sensors+self.n_motors,self.n_sensors+self.n_motors+self.n_hidden))
        self.hidden_labels = [node_labels[i] for i in self.hidden_ixs]


    def getMotorActivity(self, trial):
        '''
        Function for getting the motor activity from a system's activity
        ### THIS FUNCTION ONLY WORKS FOR SYSTEMS WITH TWO SENSORS ###
            Inputs:
                trial: int, the trial number under investigation
            Outputs:
                motor_activity: list of movements made by the animat in a trial
        '''
        motor_states = self.brain_activity[trial,:,self.n_sensors:self.n_sensors+2]
        motor_activity = []
        for state in motor_states:
            state = list(state)
            if state==[0,0] or state==[1,1]:
                motor_activity.append(0)
            elif state==[1,0]:
                motor_activity.append(1)
            else: # state==[0,1]
                motor_activity.append(-1)
        return motor_activity

    def get_phi_from_subsystem(self, state, node_indices = (4,5,6,7)):

        #calculate phi on specified subsystem,
        #4,5,6,7 all hidden nodes of the animat
        subsystem = pyphi.Subsystem(self.brain, state, node_indices)
        phi = pyphi.compute.phi(subsystem)

        return phi

    def get_phi_from_complexes(self, state):
        complexes = pyphi.compute.network.complexes(self.brain, state)
        if len(complexes) == 0:
            phi = 0.0
        elif len(complexes) > 1:
            phi = []
            for complex_num in range(len(complexes)):
                    phi.append(complexes[complex_num].phi)
        else:
            phi_table[run].append(complexes[0].phi)
        return phi

    def save_system_irreducibility_analysis(self):
        '''

            Inputs:

            Outputs:
        '''

        sias = {}
        for s in self.unique_states:
            sia = pyphi.compute.complexes(self.brain,s)
            if len(sia)==0:
                sias.update({state2num(s) : None})
            else:
                sias.update({state2num(s) : sia})
        self.sias = sias


    def save_major_complex(self):
        '''

            Inputs:

            Outputs:
        '''

        MCs = {}
        n_states = len(self.unique_states)
        n = 0
        for s in self.unique_states:
            print('Finding MC for state number {} out of {} unique states'.format(n,n_states))
            n+=1

            MC = pyphi.compute.major_complex(self.brain,s)
            if MC.phi==0:
                MCs.update({state2num(s) : None})
            else:
                MCs.update({state2num(s) : MC})
        self.MCs = MCs


    def save_phi_from_MCs(self):
        '''

            Inputs:

            Outputs:
        '''

        phis = []
        for s in self.unique_states:

            if self.MCs[state2num(s)] == None:
                phis.append(0)
            else:
                phis.append(self.MCs[state2num(s)].phi)
        self.phis = phis


    def plot_brain(self, state=None, ax=None):
        '''
        Function for plotting the brain of an animat.
        ### THIS FUNCTION ONLY WORKS WELL FOR ANIMATS WITH 8 NODES (2+2+4) ###
            Inputs:
                state: the state of the animat for plotting (alters colors to indicate activity)
                ax: for specifying which axes the graph should be plotted on

            Outputs:
                no output, just calls the actua_acency function for plotting
        '''
        ac_plot_brain(self.brain.cm, self.brain_graph, state, ax)

def create_animat(run, agent, tpm, cm, brain_activity):

    animat = Animat({})
    animat.saveBrian(tpm, cm)
    animat.saveBrainActivity(activity)
    return animat

class Block:
    '''
        THE FOLLOWING FUNCTIONS ARE MOSTLY FOR VISUALIZING OR RUNNING THE
        COMPLEXIPHI WORLD IN PYTHON (FOR CHECKING CONSISTENCY) AND ARE NOT
        WELL COMMENTED. FUNCTIONS USEFUL FOR ANALYSIS ARE COMMENTED.
    '''
    def __init__(self, size, direction, block_type, ini_x, ini_y=0):
        self.size = size
        self.direction = direction
        self.type = block_type
        self.x = ini_x
        self.y = ini_y

    def __len__(self):
        return self.size

    def set_x(self, position):
        self.x = position
    def set_y(self, position):
        self.y = position

class Screen:
    def __init__(self, width, height):
        self.screen = np.zeros((height + 1,width))
        self.width = width
        self.height = height
        self.screen_history = np.array([])

    def resetScreen(self):
        self.screen = np.zeros(self.screen.shape)
        self.screen_history = np.array([])

    def drawAnimat(self, animat):
        self.screen[-1,:] = 0
        self.screen[-1,self.wrapper(range(animat.x,animat.x+len(animat)))] = 1

    def drawBlock(self, block):
        self.screen[:-1,:] = 0
        self.screen[block.y,self.wrapper(range(block.x, block.x+len(block)))] = 1

    def saveCurrentScreen(self):
        if len(self.screen_history)==0:
            self.screen_history = copy.copy(self.screen[np.newaxis,:,:])
        else:
            self.screen_history = np.r_[self.screen_history, self.screen[np.newaxis,:,:]]

    def wrapper(self,index):
        if not hasattr(index, '__len__'):
            return index%self.width
        else:
            return [ix%self.width for ix in index]

class World:
    def __init__(self, width=16, height=35):
        self.width = width # ComplexiPhi world is 35 (and not 34! 34 is the number of updates)
        self.height = height
        self.screen = Screen(self.width, self.height)

    def _runGameTrial(self, trial, animat, block):

        total_time = self.height # 35 time steps, 34 updates
        motor_activity = animat.getMotorActivity(trial)

        # t=0 # Initial position (game hasn't started yet.)
        self.screen.resetScreen()
        self.screen.drawAnimat(animat)
        self.screen.drawBlock(block)
        self.screen.saveCurrentScreen()

        for t in range(1, total_time):

            animat.x = self.screen.wrapper(animat.x + motor_activity[t])

            if t<total_time:
                if block.direction == 'right':
                    block.x = self.screen.wrapper(block.x + 1)
                else:
                    block.x = self.screen.wrapper(block.x - 1)

                block.y = block.y + 1

            self.screen.drawAnimat(animat)
            self.screen.drawBlock(block)
            self.screen.saveCurrentScreen()
        # animat catches the block if it overlaps with it in t=34
        win = self._check_win(block, animat)

        return self.screen.screen_history, win

    def _getInitialCond(self, trial):
        animal_init_x = trial % self.width
        self.animat.set_x(animal_init_x)

        block_size = self.block_patterns[trial //(self.width * 2)]
        block_direction = 'left' if (trial // self.width) % 2 == 0 else 'right'
        block_value = 'catch' if (trial // (self.width * 2)) % 2 == 0 else 'avoid'
        block = Block(block_size, block_direction, block_value, 0)

        return self.animat, block

    def runFullGame(self, animat, block_patterns):

        if not hasattr(animat, 'brain_activity'):
            raise AttributeError("Animat needs a brain activity saved to play gameself.")
        self.animat = copy.copy(animat)
        self.block_patterns = block_patterns
        self.n_trials = self.width * 2 * len(block_patterns)

        self.history = np.zeros((self.n_trials,self.height,self.height+1,self.width))

        wins = []
        for trial in range(self.n_trials):
            self.animat, block = self._getInitialCond(trial)
            self.history[trial,:,:,:], win = self._runGameTrial(trial,self.animat, block)
            wins.append(win)
        return self.history, wins

    def get_fullgame_history(self, animat=None, block_patterns=None):
        if hasattr(self, 'history'):
            return self.history
        else:
            self.run_fullgame(animat, block_patterns)
            return self.history

    def _check_win(self, block, animat):
        block_ixs = self.screen.wrapper(range(block.x, block.x + len(block)))
        animat_ixs = self.screen.wrapper(range(animat.x, animat.x + len(animat)))
        catch = True if len(set(block_ixs).intersection(animat_ixs))>0 else False
        win = True if (block.type=='catch' and catch) or (block.type=='avoid' and not catch) else False
        return win

    def getFinalScore(self):
        score = 0
        for trial in range(self.n_trials):
            animat, block = self._getInitialCond(trial)
            # print('trial {}'.format(trial))
            # print('A0: {} B0: {} ({}, {}, {})'.format(animat.x,block.x,len(block),block.direction, block.type))

            animat.x = self.screen.wrapper(animat.x + np.sum(animat.getMotorActivity(trial)[:]))

            direction = -1 if block.direction=='left' else 1
            block.x = self.screen.wrapper(block.x + (self.height-1)*direction)

            win = 'WIN' if self._check_win(block, animat) else 'LOST'
            # print('Af: {} Bf: {}'.format(animat.x, block.x))
            # print(win)
            # print()
            score += int(self._check_win(block, animat))
        print('Score: {}/{}'.format(score, self.n_trials))
        return score
