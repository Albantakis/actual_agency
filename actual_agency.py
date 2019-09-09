import numpy as np
import numpy.random as ran
import scipy.io as sio
from scipy.stats import kde
from matplotlib import pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
import pickle
import os
import sys
import copy
import subprocess as sp
from pathlib import Path
import ipywidgets as widgets

import pyphi
import pyanimats as pa
import pyTPM as pt

# Setting colors used in plotting functions
blue, red, green, grey, white = '#77b3f9', '#f98e81', '#8abf69', '#adadad', '#ffffff'
purple, yellow = '#d279fc', '#f4db81'

### MABE RELATED FUNCTIONS
def parseTPM(TPM_jory):
    '''
    Function for parsing the output from the mabe TPMworld into a readable format
        Inputs:
            TPM_jory: (unpickled) csv output from mabe TPM-world
        Outputs:
            allgates: A list of lists (num_agents long) containing all gates in the agents genome
    '''
    start = '{'
    end = '}'
    split = r'\r\n'
    s = str(TPM_jory)
    a=True

    s_byanimats = s.split(split)[1:-1]
    allgates = []
    for animat in s_byanimats:
        gates = []
        a = True
        while a:
            idx1 = str(animat).find(start)
            idx2 = str(animat).find(end)+1
            if idx1==-1:
                a = False
            else:
                gate_string = animat[idx1:idx2]
                gates.append(eval(gate_string))
                animat = animat[idx2+1:]
        allgates.append(gates)
    return allgates

def get_genome(genomes, run, agent):
    genome = genomes[run]['GENOME_root::_sites'][agent]
    genome = np.squeeze(np.array(np.matrix(genome)))
    return genome

def getBrainActivity(data, n_agents=1, n_trials=64, n_nodes=8, n_sensors=2,n_hidden=4,n_motors=2):
    '''
    Function for generating a activity matrices for the animats given outputs from mabe
        Inputs:
            data: a pandas object containing the mabe output from activity recording
            n_agents: number of agents recorded
            n_trials: number of trials for each agent
            n_nodes: total number of nodes in the agent brain (sensors+motrs+hidden)
            n_sensors: number of sensors in the agent brain
            n_hidden: number of hidden nodes between the sensors and motors
            n_motors: number of motors in the agent brain
        Outputs:
            brain_activity: a matrix with the timeseries of activity for each trial of every agent. Dimensions(agents)
    '''
    print('Creating activity matrix from MABE output...')
    n_transitions = 34
    brain_activity = np.zeros((n_agents,n_trials,1+n_transitions,n_nodes))

    for a in list(range(n_agents)):
        for i in list(range(n_trials)):
            for j in list(range(n_transitions+1)):
                ix = a*n_trials*n_transitions + i*n_transitions + j
                if j==0:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:n_sensors]
                    hidden = np.zeros(n_hidden)
                    motor = np.zeros(n_motors)
                elif j==n_transitions:
                    sensor = np.zeros(n_sensors)
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                else:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:n_sensors]
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                nodes = np.r_[sensor, motor, hidden]
                brain_activity[a,i,j,:] = nodes
    return brain_activity

def parseActivity(path,file,n_runs=30,n_agents=61,n_trials=64,world_height=35,n_nodes=8,n_sensors=2,n_hidden=4,n_motors=2):
    '''
    Function for reading activity from MABE output of multiple runs (pkl) to array
        Inputs:
            inputs:
                path: path to where the pkl output is stored
                file: name of the file to load
                n_agents: number of agents saved (generations / steplength between saves)
                n_trials: number of trials pr agent. 2 x 16 x number of block types
                world_height: the height of the complexiphi world

        Outputs:
            outputs:
                all_activity: 4D array (runs x trials x time x nodes) containing all activity of the animat
    '''
    with open(os.path.join(path,file),'rb') as f:
        activity = pickle.load(f)

    all_activity = np.zeros((n_runs,n_agents,n_trials,world_height,n_nodes),dtype=int)
    for i in range(n_runs):
        print('{}/{}'.format(i+1,n_runs))
        all_activity[i,:,:,:,:] = getBrainActivity(activity[i],n_agents,n_trials,n_nodes,n_sensors,n_hidden,n_motors)

    with open(os.path.join(path,'activity_array.pkl'),'wb') as f:
        pickle.dump(all_activity, f)

    return all_activity


### ACTTUAL CAUSATION ANALYSIS FUNCTIONS
def get_purview(causal_link,purview_type='union'):
    '''
    This function gets the union of extended purviews of a causal link if the
    causal link has that attribute. Otherwise it gets the union of the available purviews.
        Inputs:
            inputs:
                causal_link: the list of irreducible causes of some account
                union_of_purviews: indicator if the returned value should contain the union of purviews
        Outputs:
            outputs:
                purview: the union of all purview elements across all (extended) cause purviews
    '''
    # checking if causal link has the attribute _extended_purview
    if hasattr(causal_link,'_extended_purview'):
        extended_purview = causal_link._extended_purview
    else:
        extended_purview = causal_link.purview

    if purview_type == 'union':
        if type(extended_purview) == list and len(extended_purview)>1:
            # creating the union of     purviews
            purview = set()
            for p in extended_purview:
                purview = purview.union(p)
        elif type(extended_purview) == tuple:
            purview = extended_purview

    # returning the output
    return tuple(purview)


def get_actual_causes(animat, trial, t, cause_ixs, effect_ixs):
    '''
    This function gets the irreducible causes of a transition
        Inputs:
            animat: animat object with brain activity
            trial: the trial number under investigation (int)
            t: the time of the second state in the transition (int)
            cause_ixs: the indices of the elements that may form the cause purviews in the account
            effect_ixs: the indices of the elements that constitute the occurrence under investigation
        Outputs:
            causes: the list of all causal links in the the account
    '''

    # getting the transition under investigation and defining it with pyphi
    before_state, after_state = animat.get_transition(trial,t,False)
    transition = pyphi.actual.Transition(animat.brain, before_state, after_state, cause_ixs, effect_ixs)

    # calculating the causal account and picking out the irreducible causes
    account = pyphi.actual.account(transition, direction=pyphi.Direction.CAUSE)
    causes = account.irreducible_causes

    # returning output
    return causes


def backtrack_cause(animat, trial, t, ocurrence_ixs=None, max_backsteps=3, purview_type='union', debug=False):
    '''
    Function for tracking the causes of an occurence back in time
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            t: the time of the second state in the transition (int)
            occurence_ixs: the indices of the elements that constitute the occurrence under investigation
            max_backsteps: the maximum number of steps we track the causes back
            purview_type: name of the type of purview we use to track the causes
        Outputs:
            outputs: list of lists containing all cause purviews in the causal chain
    '''

    if not hasattr(animat,'node_labels'):
        ### the following is specially designed for the analysis of Juel et al 2019 and should be generalized
        if ocurrence_ixs==None:
            ocurrence_ixs = [2,3] if animat.n_nodes==8 else [1,2] # motor ixs
        if animat.n_nodes==8:
            cause_ixs = [0,1,4,5,6,7]
            S1, S2, M1, M2, A, B, C, D = range(8)
            label_dict = {key:x for key,x in zip(range(8),['S1', 'S2', 'M1', 'M2', 'A', 'B', 'C', 'D'])}
        else:
            cause_ixs = [0,3,4,5,6]
            S1, M1, M2, A, B, C, D = range(7)
            label_dict = {key:x for key,x in zip(range(7),['S1', 'M1', 'M2', 'A', 'B', 'C', 'D'])}
    else:
        cause_ixs = list(animat.sensor_ixs) + list(animat.hidden_ixs)
        if ocurrence_ixs==None:
            ocurrence_ixs = animat.motor_ixs

        if debug:
            print('MAKE A PROPER label_dict FOR RUNNING DEBUG')

    causal_chain = []

    backstep = 1
    end = False
    effect_ixs = ocurrence_ixs
    while not end and backstep <= max_backsteps and t>0:

        causes = get_actual_causes(animat, trial, t, cause_ixs, effect_ixs)
        n_causal_links = len(causes)

        if n_causal_links==0:
            end=True

        # use the union of the purview of all actual causes as the next ocurrence (effect_ixs) in the backtracking
        effect_ixs = [p for cause in causes for p in get_purview(cause,purview_type)]
        effect_ixs = list(set(effect_ixs))

        if not hasattr(animat,'node_labels'):
            if animat.n_nodes==8:
                if (len(effect_ixs)==1 and (S1 in effect_ixs or S2 in effect_ixs)):
                    end=True
                elif (len(effect_ixs)==2 and (S1 in effect_ixs and S2 in effect_ixs)):
                    end=True
            else:
                if (len(effect_ixs)==1 and (S1 in effect_ixs)):
                    end=True
        else:
            if all([i in animat.sensor_labels for i in effect_ixs]):
                end=True

        if debug:
            print(f't: {t}')
            print_transition(animat.get_transition(trial,t))
            print(causes)
            next_effect = [label_dict[ix] for ix in effect_ixs]
            print('Next effect_ixs: {}'.format(next_effect))

        causal_chain.append(causes)
        t -= 1
        backstep += 1
        if t==-1:
            print('t=-1 reached.')
    return causal_chain


def backtrack_cause_trial(animat,trial,max_backsteps=3,ocurrence_ixs=None,purview_type='union'):
    '''
    Calculates the causal chain leading to an occurence
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            max_backsteps: the maximum number of steps we track the causes back (int). also the first timestep tracked back
            occurence_ixs: the indices of the elements that constitute the occurrence under investigation
            purview_type: name of the type of purview we use to track the causes
        Outputs:
            causal_chain: a list of backtracking patterns for each timestep in a trial
    '''

    causal_chain = []
    n_times = animat.brain_activity.shape[1]

    if ocurrence_ixs is None:
        occurence_ixs = animat.motor_ixs

    print('Calculating causal chain for trial {}.'.format(trial))
    aux = ran.rand()
    if aux<0.02:
        print('Have patience young padawan!')
    elif aux<0.04:
        print('have faith! It will finish eventually...')
    elif aux<0.05:
        print("this is a chicken, for your entertainment      (  ')>  ")
    elif aux<0.06:
        print('This might be a good time for a coffee')

    for t in range(max_backsteps,n_times):
        causal_chain.append(backtrack_cause(animat, trial, t, ocurrence_ixs, max_backsteps, purview_type))

    return causal_chain



def calc_causal_history(animat, trial, only_motor=True,debug=False):
    '''
    Calculates animat's direct cause history, defined as the direct causes of
    every transition (only motor or not) across a trial.
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            only_motor: indicates whether the occurrence under investigation is only motors or the wholde network
        Outputs:
            direct_cause_history: list of lists of irreducible cause purviews
    '''
    if not hasattr(animat,'node_labels'):
        ### the following is specially designed for the analysis of Juel et al 2019 and should be generalized
        if animat.n_nodes==8:
            cause_ixs = [0,1,4,5,6,7]
            effect_ixs = [2,3] if only_motor else [2,3,4,5,6,7]
        else:
            cause_ixs = [0,3,4,5,6]
            effect_ixs = [1,2] if only_motor else [1,2,3,4,5,6]
    else:
        cause_ixs = animat.sensor_ixs + animat.hidden_ixs
        effect_ixs = animat.motor_ixs if only_motor else animat.motor_ixs+animat.hidden_ixs

    direct_cause_history = []
    n_times = animat.brain_activity.shape[1]

    for t in reversed(range(1,n_times)):

        before_state, after_state = animat.get_transition(trial,t,False)
        transition = pyphi.actual.Transition(animat.brain, before_state, after_state, cause_ixs, effect_ixs)
        account = pyphi.actual.account(transition, direction=pyphi.Direction.CAUSE)
        causes = account.irreducible_causes

        if debug:
            print(f't: {t}')
            print_transition((before_state,after_state))
            print(causes)

        direct_cause_history.append(causes)

    return direct_cause_history


def get_alpha_cause_account_distribution(cause_account, n_nodes, animat):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    if animat is not None:
        n_nodes = animat.n_nodes

    alpha_dist = np.zeros(n_nodes)
    for causal_link in cause_account:
        # checking if the causal link has an extended purview
        if hasattr(causal_link,'_extended_purview'):
            ext_purv = causal_link._extended_purview
            # getting the alpha and the number of purviews over which it should be divided
            alpha = causal_link.alpha
            n_purviews = len(ext_purv)
            alpha = alpha/n_purviews

            # looping over purviews and dividing alpha to nodes
            for purview in ext_purv:
                purview_length = len(purview)
                alpha_dist[list(purview)] += alpha/purview_length
        else:
            purview = list(causal_link.purview)
            alpha = causal_link.alpha
            purview_length = len(purview)
            alpha_dist[list(purview)] += alpha/purview_length

    if animat is None:
        alpha_dist = alpha_dist[[0,3,4,5,6]] if n_nodes==7 else alpha_dist[[0,1,4,5,6,7]]
    else:
        alpha_dist = alpha_dist[animat.sensor_ixs+animat.hidden_ixs]

    return alpha_dist


def get_backtrack_array(causal_chain,n_nodes,animat=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    n_backsteps = len(causal_chain)
    if animat is None:
        BT = np.zeros((n_backsteps,n_nodes-2))
    else:
        BT = np.zeros((n_backsteps,animat.n_nodes-animat.n_motors))


    for i, cause_account in enumerate(causal_chain):
        BT[n_backsteps - (i+1),:] = get_alpha_cause_account_distribution(cause_account, n_nodes, animat)

    return BT

def get_causal_history_array(causal_chain,n_nodes,mode='alpha'): # OLD
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    n_timesteps = len(causal_chain)
    causal_history = np.zeros((n_timesteps,n_nodes))
    for i in range(n_timesteps):
        for causal_link in causal_chain[i]:
            if mode=='alpha':
                weight = causal_link.alpha
            else:
                weight = 1
            causal_history[n_timesteps - (i+1),list(get_purview(causal_link))] += weight
    return causal_history


def get_occurrences(activityData,numSensors,numHidden,numMotors):
    '''
    Function for converting activity data from mabe to past and current occurences.
        Inputs:
            activityData: array containing all activity data to be converted ((agent x) trials x time x nodes)
            numSensors: number of sensors in the agent brain
            numHidden: number of hiden nodes in the agent brain
            numMotors: number of motor units in the agent brain
        Outputs:
            x: past occurences (motor activity set to 0, since they have no effect on the future)
            y: current occurences (sensor activity set to 0, since they are only affected by external world)
    '''
    size = activityData.shape
    x = np.zeros(size)
    y = np.zeros(size)


    if len(size)==4:
        # deleting one timestep from each trial
        x = np.delete(x,(-1),axis=2)
        y = np.delete(y,(-1),axis=2)

        # filling matrices with values
        x = copy.deepcopy(activityData[:,:,:-1,:])
        y = copy.deepcopy(activityData[:,:,1:,:])

        # setting sensors to 0 in y, and motors to zeros in x
        x[:,:,:,numSensors:numSensors+numMotors] = np.zeros(x[:,:,:,numSensors:numSensors+numMotors].shape)
        y[:,:,:,:numSensors] = np.zeros(y[:,:,:,:numSensors].shape)

    elif len(size)==3:
        # deleting one timestep from each trial
        x = np.delete(x,(-1),axis=1)
        y = np.delete(y,(-1),axis=1)

        # filling matrices with values
        x = copy.deepcopy(activityData[:,:-1,:])
        y = copy.deepcopy(activityData[:,1:,:])

        # setting sensors to 0 in y, and motors to zeros in x
        x[:,:,numSensors:numSensors+numMotors] = np.zeros(x[:,:,numSensors:numSensors+numMotors].shape)
        y[:,:,:numSensors] = np.zeros(y[:,:,:numSensors].shape)

    return x, y

def get_all_unique_transitions(activityData,numSensors=2,numHidden=4,numMotors=2):

    x,y = get_occurrences(activityData,numSensors,numHidden,numMotors)

    trials, times, nodes = x.shape

    unique = []
    transition_number = []

    for tr in range(trials):
        for t in range(times):
            transition = (x[tr][t][:], y[tr][t][:])

            trnum_curr = state2num(list(x[tr][t][:]) + (list(y[tr][t][:])))

            if trnum_curr not in transition_number:
                unique.append(transition)

            transition_number.append(trnum_curr)
    nums = np.array(transition_number).reshape(trials,times).astype(int).tolist()
    return unique, nums



def AnalyzeTransitions(network, activity, cause_indices=[0,1,4,5,6,7], effect_indices=[2,3],
                       sensor_indices=[0,1], motor_indices=[2,3],
                       purview = [],alpha = [],motorstate = [],transitions = [], account = []):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    states = len(activity)
    n_nodes = len(activity[0])
    x_indices = [i for i in range(n_nodes) if i not in motor_indices]
    y_indices = [i for i in range(n_nodes) if i not in sensor_indices]

    if len(transitions)>0:
        tran = [np.append(transitions[i][0][x_indices],transitions[i][1][y_indices]) for i in list(range(0,len(transitions)))]
    else:
        tran = []

    for s in list(range(states-1)):
        # 2 sensors
        x = activity[s,:].copy()
        x[motor_indices] = [0]*len(motor_indices)
        y = activity[s+1,:].copy()
        y[sensor_indices] = [0]*len(sensor_indices)

        occurence = np.append(x[x_indices],y[y_indices]).tolist()

        # checking if this transition has never happened before for this agent
        if not any([occurence == t.tolist() for t in tran]):
            # generating a transition
            transition = pyphi.actual.Transition(network, x, y, cause_indices,
                effect_indices, cut=None, noise_background=False)
            CL = transition.find_causal_link(pyphi.Direction.CAUSE, tuple(effect_indices), purviews=False, allow_neg=False)
            AA = pyphi.actual.account(transition,pyphi.Direction.CAUSE)


            alpha.append(CL.alpha)
            purview.append(CL.purview)
            motorstate.append(tuple(y[motor_indices]))
            account.append(AA)


            # avoiding recalculating the same occurence twice
            tran.append(np.array(occurence))
            transitions.append([np.array(x),np.array(y)])

    return purview, alpha, motorstate, transitions, account

def createPandasFromACAnalysis(LODS,agents,activity,TPMs,CMs,labs,
                               cause_indices=[0,1,4,5,6,7], effect_indices=[2,3],
                               sensor_indices=[0,1], motor_indices=[2,3]):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    catch = []
    purview = []
    alpha = []
    motor = []
    transitions = []
    account = []

    for lod in LODS:
        purview_LOD = []
        alpha_LOD = []
        motor_LOD = []
        transitions_LOD = []
        account_LOD = []
        catch_LOD = []

        for agent in agents:
            print('LOD: {} out of {}'.format(lod,np.max(LODS)))
            print('agent: {} out of {}'.format(agent,np.max(agents)))
            purview_agent = []
            alpha_agent = []
            motor_agent = []
            transitions_agent = []
            account_agent = []
            catch_agent = []

            tr = []
            TPM = np.squeeze(TPMs[lod,agent,:,:])
            CM = np.squeeze(CMs[lod,agent,:,:])
            TPMmd = pyphi.convert.to_multidimensional(TPM)
            network_2sensor = pyphi.Network(TPMmd, cm=CM, node_labels=labs)

            for t in range(64):
                purview_agent, alpha_agent, motor_agent, transitions_agent, account_agent = AnalyzeTransitions(
                    network_2sensor, np.squeeze(activity[lod,agent,t,:,:]),
                    purview = purview_agent, alpha = alpha_agent, account = account_agent,
                    motorstate = motor_agent, transitions=transitions_agent,
                    cause_indices=cause_indices, effect_indices=effect_indices,
                    sensor_indices=sensor_indices, motor_indices=motor_indices)
                catch_agent.append(1) if t<32 else catch.append(0)

            purview_LOD.append(purview_agent)
            alpha_LOD.append(alpha_agent)
            motor_LOD.append(motor_agent)
            transitions_LOD.append(transitions_agent)
            account_LOD.append(account_agent)
            catch_LOD.append(catch_agent)


        purview.append(purview_LOD)
        alpha.append(alpha_LOD)
        motor.append(motor_LOD)
        transitions.append(transitions_LOD)
        account.append(account_LOD)
        catch.append(catch_LOD)

    purview_aux = []
    alpha_aux = []
    motor_aux = []
    transitions_aux = []
    account_aux = []
    lod_aux = []
    agent_aux = []
    catch_aux = []
    s1 = []
    s2 = []
    h1 = []
    h2 = []
    h3 = []
    h4 = []
    hiddenInPurview = []
    sensorsInPurview = []

    idx = 0

    for lod in list(range(0,len(LODS))):
        for agent in list(range(0,len(agents))):
            for i in list(range(len(purview[lod][agent]))):

                motor_aux.append(np.sum([ii*(2**idx) for ii,idx in zip(motor[lod][agent][i],list(range(0,len(motor[lod][agent][i]))))]))
                transitions_aux.append(transitions[lod][agent][i])
                account_aux.append(account[lod][agent][i])
                lod_aux.append(lod)
                agent_aux.append(agent)
                catch_aux.append(catch)

                if purview[lod][agent][i] is not None:
                    purview_aux.append([labs_2sensor[ii] for ii in purview[lod][agent][i]])
                    s1.append(1 if 's1' in purview_aux[idx] else 0)
                    s2.append(1 if 's2' in purview_aux[idx] else 0)
                    h1.append(1 if 'h1' in purview_aux[idx] else 0)
                    h2.append(1 if 'h2' in purview_aux[idx] else 0)
                    h3.append(1 if 'h3' in purview_aux[idx] else 0)
                    h4.append(1 if 'h4' in purview_aux[idx] else 0)
                    alpha_aux.append(alpha[lod][agent][i])
                    idx+=1

                else:
                    purview_aux.append('none')
                    alpha_aux.append(alpha[lod][agent][i])
                    s1.append(0)
                    s2.append(0)
                    h1.append(0)
                    h2.append(0)
                    h3.append(0)
                    h4.append(0)
                    idx+=1

                hiddenInPurview.append(h1[idx-1]+h2[idx-1]+h3[idx-1]+h4[idx-1])
                sensorsInPurview.append(s1[idx-1]+s2[idx-1])

    dictforpd = {'purview':purview_aux,
                    'motor':motor_aux,
                    'alpha':alpha_aux,
                    's1':s1,
                    's2':s2,
                    'h1':h1,
                    'h2':h2,
                    'h3':h3,
                    'h4':h4,
                    'hiddenInPurview':hiddenInPurview,
                    'sensorsInPurview':sensorsInPurview,
                    'catch': catch_aux,
                    'transition': transitions_aux,
                    'account': account_aux,
                    'LOD': lod_aux,
                    'agent': agent_aux,
                    }

    panda = pd.DataFrame(dictforpd)

    return panda




### DATA ANALYSIS FUNCTIONS

def Bootstrap_mean(data,n):
    '''
    Function for doing bootstrap resampling of the mean for a 2D data matrix.
        Inputs:
            data: raw data samples to be bootsrap resampled (samples x datapoints)
            n: number of bootstrap samples to draw
        Outputs:
            means: matrix containing all bootstrap samples of the mean (n x datapoints)
    '''
    datapoints = len(data)
    timesteps = len(data[0])

    idx = list(range(n))
    means = [0 for i in idx]
    for i in idx:
        # drawing random timeseries (with replacement) from data
        bootstrapdata = np.array([data[d][:] for d in ran.choice(list(range(0,datapoints)),datapoints,replace=True)])
        means[i] = np.nanmean(bootstrapdata,0)

    return means

def get_bootstrap_stats(data,n=500):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    fit = Bootstrap_mean(data,n)
    return np.mean(fit,0), np.std(fit,0)




### PLOTTING FUNCTIONS

def plot_ACanalysis(animat, world, trial, t, causal_chain=None, plot_state_history=False,
                    causal_history=None,causal_history_motor=None, plot_causal_account=True,
                    plot_state_transitions=True, n_backsteps=None):

    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    n_sensors = animat.n_sensors
    n_nodes = animat.n_nodes
    # PLOT SCREEN
    _, block = world._get_initial_condition(trial)
    fullgame_history = world.get_fullgame_history()
    wins = world.wins

    n_cols = 4
    if plot_state_history:
        n_cols +=1
    if causal_history is not None:
        n_cols +=1
    if causal_history_motor is not None:
        n_cols +=1
    if causal_chain is not None:
        n_cols +=1

    n_rows = 2
    col=0

    plt.subplot2grid((2,n_cols),(0,col),rowspan=2, colspan=2)
    col+=2
    plt.imshow(fullgame_history[trial,t,:,:],cmap=plt.cm.binary);
    plt.xlabel('x'); plt.ylabel('y')
    win = 'WON' if wins[trial] else 'LOST'
    direction = '━▶' if block.direction=='right' else '◀━'
    plt.title('Game - Trial: {}, {}, {}, {}'.format(block.size,block.type,direction,win))

    # PLOT CAUSAL HISTORIES
    labels = ['S1', 'S2', 'M1', 'M2', 'A', 'B', 'C', 'D'] if n_sensors==2 else ['S1', 'M1', 'M2', 'A', 'B', 'C', 'D']

    if plot_state_history:
        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        # plt.colorbar(shrink=0.2)
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node'); plt.ylabel('Time')
        plt.title('Brain states history')
        plt.imshow(animat.brain_activity[trial],cmap=plt.cm.binary)
        plt.colorbar(shrink=0.1)
        plt.axhline(t-0.5,color=red)
        plt.axhline(t+0.5,color=red)

    if causal_history is not None:
        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        plt.imshow(causal_history, cmap=plt.cm.magma_r)
        plt.colorbar(shrink=0.2)
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node'); plt.ylabel('Time')
        plt.title('Causal history')
    if causal_history_motor is not None:
        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        plt.imshow(causal_history_motor, vmax=6, cmap=plt.cm.magma_r)
        plt.colorbar(shrink=0.2)
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node')
        plt.title('Causal history of M1,M2')

    # PLOT BACKTRACKING
    if causal_chain is not None:
        BT = get_backtrack_array(causal_chain,n_nodes=n_nodes)
        if n_backsteps is None:
            n_backsteps = len(causal_chain)
        BT = BT[-n_backsteps:]
        S = np.zeros((world.height,n_nodes))
        ixs = [0,3,4,5,6] if n_sensors==1 else [0,1,4,5,6,7]
        S[t-n_backsteps:t,ixs] = BT

        plt.subplot2grid((2,n_cols),(0,col),rowspan=2)
        col+=1
        plt.imshow(S,vmin=0,vmax=np.max(BT),cmap=plt.cm.magma_r)
        plt.colorbar(shrink=0.2)

        labels = ['S1', 'S2', 'M1', 'M2', 'A', 'B', 'C', 'D'] if n_sensors==2 else ['S1', 'M1', 'M2', 'A', 'B', 'C', 'D']
        plt.xticks(range(n_nodes),labels)
        plt.xlabel('Node')
        plt.title('Backtracking of M1,M2')

    # PLOT ANIMAT BRAIN
    if plot_state_transitions:
        transition = animat.get_transition(trial,t,False)
        if animat.n_nodes==8:
            cause_ixs = [0,1,4,5,6,7]
            effect_ixs = [2,3]
        else:
            cause_ixs = [0,3,4,5,6]
            effect_ixs = [1,2]
        T = pyphi.actual.Transition(animat.brain, transition[0], transition[1], cause_ixs, effect_ixs)
        account = pyphi.actual.account(T, direction=pyphi.Direction.CAUSE)
        causes = account.irreducible_causes

        plt.subplot2grid((2,n_cols),(0,col),colspan=2)
        animat.plot_brain(transition[0])
        plt.title('Brain\n\nT={}'.format(t-1),y=0.85)

        if plot_causal_account:
            plt.text(0,0,causes,fontsize=12)


        plt.subplot2grid((2,n_cols),(1,col),colspan=2)
        animat.plot_brain(transition[1])
        plt.title('T={}'.format(t),y=0.85)

        plt.text(-2,6,transition_str(transition),fontsize=12)

    else:
        state = animat.get_state(trial,t)
        plt.subplot2grid((18,n_cols),(0,col),colspan=2,rowspan=9)
        animat.plot_brain(state)
        plt.subplot2grid((18,n_cols),(9,col),colspan=2)
        plt.imshow(np.array(state)[np.newaxis,:],cmap=plt.cm.binary)
        plt.yticks([])
        plt.xticks(range(animat.n_nodes),labels)
    # plt.tight_layout()

def plot_brain(cm, graph=None, state=None, ax=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    n_nodes = cm.shape[0]
    if n_nodes==7:
        labels = ['S1','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), #'S2': (20, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,1,1,2,2,2,2)

        ini_hidden = 3

    elif n_nodes==8:
        labels = ['S1','S2','M1','M2','A','B','C','D']
        pos = {'S1': (5,40), 'S2': (15, 40),
           'A': (0, 30), 'B': (20, 30),
           'C': (0, 20), 'D': (20, 20),
          'M1': (5,10), 'M2': (15,10)}
        nodetype = (0,0,1,1,2,2,2,2)
        ini_hidden = 4

    if graph is None:
        graph = nx.from_numpy_matrix(cm, create_using=nx.DiGraph())
        mapping = {key:x for key,x in zip(range(n_nodes),labels)}
        graph = nx.relabel_nodes(graph, mapping)

    state = [1]*n_nodes if state==None else state

    blue, red, green, grey, white = '#6badf9', '#f77b6c', '#8abf69', '#adadad', '#ffffff'
    blue_off, red_off, green_off, grey_off = '#e8f0ff','#ffe9e8', '#e8f2e3', '#f2f2f2'

    colors = np.array([red, blue, green, grey, white])
    colors = np.array([[red_off,blue_off,green_off, grey_off, white],
                       [red,blue,green, grey, white]])

    node_colors = [colors[state[i],nodetype[i]] for i in range(n_nodes)]
    # Grey Uneffective or unaffected nodes
    cm_temp = copy.copy(cm)
    cm_temp[range(n_nodes),range(n_nodes)]=0
    unaffected = np.where(np.sum(cm_temp,axis=0)==0)[0]
    uneffective = np.where(np.sum(cm_temp,axis=1)==0)[0]
    noeffect = list(set(unaffected).union(set(uneffective)))
    noeffect = [ix for ix in noeffect if ix in range(ini_hidden,ini_hidden+4)]
    node_colors = [node_colors[i] if i not in noeffect else colors[state[i],3] for i in range(n_nodes)]

    #   White isolate nodes
    isolates = [x for x in nx.isolates(graph)]
    node_colors = [node_colors[i] if labels[i] not in isolates else colors[0,4] for i in range(n_nodes)]

    self_nodes = [labels[i] for i in range(n_nodes) if cm[i,i]==1]
    linewidths = [2.5 if labels[i] in self_nodes else 1 for i in range(n_nodes)]

#     fig, ax = plt.subplots(1,1, figsize=(4,6))
    nx.draw(graph, with_labels=True, node_size=800, node_color=node_colors,
    edgecolors='#000000', linewidths=linewidths, pos=pos, ax=ax)


def plot_mean_with_errors(x, y, yerr, color, label=None, linestyle=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    if len(yerr)==2: # top and bottom percentiles
        plt.fill_between(x, yerr[0], yerr[1], color=color, alpha=0.1)
    else:
        plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.1)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle)


def plot_LODdata_and_Bootstrap(x,LODdata,label='data',color='b',linestyle='-',figsize=[20,10]):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    fit = Bootstrap_mean(LODdata,500)
    m_fit = np.mean(fit,0)
    s_fit = np.std(fit,0)
    fig = plt.figure(figsize=figsize)
    for LOD in LODdata:
        plt.plot(x,LOD,'r',alpha=0.1)
    plt.fill_between(x, m_fit-s_fit, m_fit+s_fit, color=color, alpha=0.2)
    plt.plot(x, m_fit, label=label, color=color, linestyle=linestyle)

    return fig



def plot_2LODdata_and_Bootstrap(x,LODdata1,LODdata2,label=['data1','data2'],color=['k','y'],linestyle='-',figsize=[20,10],fig=None,subplot=111):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    fit1 = Bootstrap_mean(LODdata1,500)
    m_fit1 = np.mean(fit1,0)
    s_fit1 = np.std(fit1,0)
    fit2 = Bootstrap_mean(LODdata2,500)
    m_fit2 = np.mean(fit2,0)
    s_fit2 = np.std(fit2,0)
    if fig==None:
        fig = plt.figure(figsize=figsize)
    plt.subplot(subplot)
    for LOD1,LOD2 in zip(LODdata1,LODdata2):
        plt.plot(x,LOD1,color[0],alpha=0.1)
        plt.plot(x,LOD2,color[1],alpha=0.1)
    plt.fill_between(x, m_fit1-s_fit1, m_fit1+s_fit1, color=color[0], alpha=0.2)
    plt.plot(x, m_fit1, label=label[0], color=color[0], linestyle=linestyle)
    plt.fill_between(x, m_fit2-s_fit2, m_fit2+s_fit2, color=color[1], alpha=0.2)
    plt.plot(x, m_fit2, label=label[1], color=color[1], linestyle=linestyle)

    return fig


def hist2d_2LODdata(LODdata1x,LODdata1y,LODdata2x,LODdata2y, nbins=20):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    xmin = np.min((np.min(LODdata1x),np.min(LODdata2x)))
    xmax = np.max((np.max(LODdata1x),np.max(LODdata2x)))
    ymin = np.min((np.min(LODdata1y),np.min(LODdata2y)))
    ymax = np.max((np.max(LODdata1y),np.max(LODdata2y)))

    xbins = np.linspace(xmin,xmax,nbins)
    ybins = np.linspace(ymin,ymax,nbins)
    plt.figure()
    plt.subplot(121)
    plt.hist2d(np.ravel(LODdata1x),np.ravel(LODdata1y),[xbins,ybins],norm=mpl.colors.LogNorm())
    plt.subplot(122)
    plt.hist2d(np.ravel(LODdata2x),np.ravel(LODdata2y),[xbins,ybins],norm=mpl.colors.LogNorm())

def plot_mean_with_errors(x, y, yerr, color, label=None, linestyle=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    plt.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.1)
    plt.plot(x, y, label=label, color=color, linestyle=linestyle)

def plot_2Ddensity(x,y, plot_samples=True, cmap=plt.cm.Blues, color=None, markersize=0.7):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    data = np.c_[x,y]
    k = kde.gaussian_kde(data.T)
    nbins = 20
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap)

    plt.plot(x,y,'.', color=color, markersize=markersize)


### OTHER FUNCTIONS

def state_str(state):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    if len(state)==8:
        s = '{}|{}|{}'.format(state[:2],state[2:4],state[4:])
    elif len(state)==7:
        s = '{}|{}|{}'.format(state[:1],state[1:3],state[3:])
    else:
        raise Exception('State of length {} is not accepted.'.format(len(state)))
    return s

def transition_str(transition):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    state1, state2 = transition
    s = state_str(state1)+' ━━▶'+state_str(state2)
    return s

def print_state(state):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    if len(state)==8:
        s = '   S      M        H\n' + state_str(state)

    else:
        s = '  S     M        H\n' + state_str(state)
    print(s)

def print_transition(transition):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    state1, state2 = transition
    if len(state1)==8:
        print('   S      M        H                S     M        H\n' + state_str(state1)+' ━━▶'+state_str(state2))
    else:
        print('  S     M        H               S     M        H\n' + state_str(state1)+' ━━▶'+state_str(state2))

def get_event_id(task,n_sensors,run,agent,trial=None,t=None):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    if t!=None:
        return '_'.join(['task',str(task),'sensor',str(n_sensors),'run',str(run),'agent',str(agent),'trial',str(trial),'t',str(t)])
    if trial!=None:
        return '_'.join(['task',str(task),'sensor',str(n_sensors),'run',str(run),'agent',str(agent),'trial',str(trial)])
    else:
        return '_'.join(['task',str(task),'sensor',str(n_sensors),'run',str(run),'agent',str(agent)])

def load_dataset(path):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    print(os.listdir(path))

    data = []
    with open(os.path.join(path,'genome.pkl'),'rb') as f:
        genomes = pickle.load(f)
        data.append(genomes)
    with open(os.path.join(path,'LOD_data.pkl'),'rb') as f:
        LOD_data = pickle.load(f)
        data.append(LOD_data)
    if os.path.isfile(os.path.join(path,'activity_array.pkl')):
        with open(os.path.join(path,'activity_array.pkl'),'rb') as f:
            activity = pickle.load(f)
            data.append(activity)
    if os.path.isfile(os.path.join(path,'fullTPM.pkl')):
        with open(os.path.join(path,'fullTPM.pkl'),'rb') as f:
            TPMs = pickle.load(f)
            data.append(TPMs)
    if os.path.isfile(os.path.join(path,'CM.pkl')):
        with open(os.path.join(path,'CM.pkl'),'rb') as f:
            CMs = pickle.load(f)
            data.append(CMs)
    if os.path.isfile(os.path.join(path,'inferred_CM.pkl')):
        with open(os.path.join(path,'inferred_CM.pkl'),'rb') as f:
            inferred_CMs = pickle.load(f)
            data.append(inferred_CMs)
    return tuple(data)

def state2num(state):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    # send in a state as a list
    num = 0
    for i in range(len(state)):
        num += (2**i)*state[-(i+1)]

    # returns the number associated with the state
    return num

def num2state(num,n_nodes):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''

    number = '{0:0' + str(n_nodes) + 'b}'
    state = number.format(num)
    state = [int(i) for i in state]

    # returns the state
    return state

def get_unique_states(activity):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    statenum = []
    for trial in activity:
        for state in trial:
            statenum.append(int(state2num(state)))

    uniques = list(set(statenum))

    states = []
    for n in uniques:
        states.append(num2state(n,activity.shape[2]))

    nums = np.array(statenum).reshape(activity.shape[0],activity.shape[1]).astype(int).tolist()

    return states, nums


def load_data(path,experiment_list,n_trials=128,file_names=['genome.pkl','activity.pkl','LOD_data.pkl'],
              gate_types=['deterministic','decomposable'], animat_params={}):

    # defining dataframe
    cols = ['Experiment','Run','agent',
        'n_nodes','n_sensor','n_motor','n_hidden',
        'unique transitions','unique states',
        'TPM','CM','connected nodes','fitness',
        'max Phi','mean Phi','max distinctions','mean distinctions',
        'DC purview length','DC total alpha','DC hidden ratio',
        'CC length','DC total alpha','DC hidden ratio']
    df = pd.DataFrame([],columns = cols)

    # looping over all experiments
    exp_n = 0
    for exp in experiment_list:

        print('loading ' + exp)
        # loading data
        with open(path+'/'+exp+'/'+file_names[0],'rb') as f:
            all_genomes = pickle.load(f)
        # and activity
        with open(path+'/'+exp+'/'+file_names[1],'rb') as f:
            activity = pickle.load(f)
        # and LOD data
        with open(path+'/'+exp+'/'+file_names[2],'rb') as f:
            LODs = pickle.load(f)

        n_runs = len(all_genomes)
        n_agents = len(all_genomes[0])

        # going through all runs
        for r in range(n_runs):
            print('run #' + str(r))

            # reformat the activity to a single list for each trial
            brain_activity = getBrainActivity(activity[r], n_agents,n_trials=n_trials)

            # get number of nodes
            n_hidden = (len(activity[r]['hidden_LIST'][0])+1)//2
            n_motors = (len(activity[r]['output_LIST'][0])+1)//2
            n_sensors = (len(activity[r]['input_LIST'][0])+1)//2
            n_nodes = n_hidden+n_sensors+n_motors

            # going through all agents
            for a in range(n_agents):

                # new row to be added to df
                new_row = {}

                # get genome
                genome = get_genome(all_genomes, r, a)

                # parsing TPM, CM
                TPM, CM = pt.genome2TPM_combined(genome,n_nodes, n_sensors, n_motors, gate_types[exp_n])

                # pick out activity for agent
                BA = brain_activity[a]

                # defining the animat
                animat = pa.Animat(animat_params)
                animat.saveBrainActivity(BA)
                animat.saveBrain(TPM,CM)

                # Find unique transitions and states
                saveUniqueState()
                #transitions, ids = animat.get_unique_transitions() # DOES NOT WORK NOW
                transitions = 'TBD'

                # calculating fitness
                fitness = LODs[r]['correct_AVE'][a]/(LODs[r]['correct_AVE'][a]+LODs[r]['incorrect_AVE'][a])
                if df.shape[0] ==0:
                    df = pd.DataFrame({'Experiment' : exp,
                           'Run' : r,
                           'Agent' : a,
                           'animat' : animat,
                           'fitness' : fitness,
                           'n_nodes' : n_nodes,
                           'n_sensor' : n_sensors,
                           'n_motor' : n_motors,
                           'n_hidden' : n_hidden,
                           'connected nodes' : sum(np.sum(CM,0)*np.sum(CM,1)>0),
                           'max Phi' : ['TBD'],
                           'mean Phi' : ['TBD'],
                           'max distinctions' : ['TBD'],
                           'mean distinctions' : ['TBD'],
                           'DC purview length' : ['TBD'],
                           'DC total alpha' : ['TBD'],
                           'DC hidden ratio' : ['TBD'],
                           'CC length' : ['TBD'],
                           'DC total alpha' : ['TBD'],
                           'DC hidden ratio' : ['TBD']
                            })
                else:
                    df2 = pd.DataFrame({'Experiment' : exp,
                           'Run' : r,
                           'Agent' : a,
                           'animat' : animat,
                           'fitness' : fitness,
                           'n_nodes' : n_nodes,
                           'n_sensor' : n_sensors,
                           'n_motor' : n_motors,
                           'n_hidden' : n_hidden,
                           'connected nodes' : sum(np.sum(CM,0)*np.sum(CM,1)>0),
                           'max Phi' : ['TBD'],
                           'mean Phi' : ['TBD'],
                           'max distinctions' : ['TBD'],
                           'mean distinctions' : ['TBD'],
                           'DC purview length' : ['TBD'],
                           'DC total alpha' : ['TBD'],
                           'DC hidden ratio' : ['TBD'],
                           'CC length' : ['TBD'],
                           'DC total alpha' : ['TBD'],
                           'DC hidden ratio' : ['TBD']
                            })
                    df = df.append(df2)
    df2 = df.set_index(['Experiment','Run','Agent'])
    return df2
