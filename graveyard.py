### GRAVEYARD
### WHERE FUNCTIONS COME TO DIE
### BUT ARE KEPT ALIVE FOR BACKWARDS COMPATIBILITY

import numpy as np
import numpy.random as ran
import scipy.io as sio
from scipy.stats import kde
import networkx as nx
import pandas as pd
import pickle
import os
import sys
import copy
import subprocess as sp
from pathlib import Path
import ipywidgets as widgets
import math

import pyphi
import pyanimats as pa
import pyTPM as pt

def get_alpha_cause_account_distribution(cause_account, n_nodes, animat=None):
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


def get_occurrences(activityData,numSensors,numHidden,numMotors):
    '''
    Function for converting activity data from mabe to past and current occurrences.
        Inputs:
            activityData: array containing all activity data to be converted ((agent x) trials x time x nodes)
            numSensors: number of sensors in the agent brain
            numHidden: number of hiden nodes in the agent brain
            numMotors: number of motor units in the agent brain
        Outputs:
            x: past occurrences (motor activity set to 0, since they have no effect on the future)
            y: current occurrences (sensor activity set to 0, since they are only affected by external world)
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

        occurrence = np.append(x[x_indices],y[y_indices]).tolist()

        # checking if this transition has never happened before for this agent
        if not any([occurrence == t.tolist() for t in tran]):
            # generating a transition
            transition = pyphi.actual.Transition(network, x, y, cause_indices,
                effect_indices, cut=None, noise_background=False)
            CL = transition.find_causal_link(pyphi.Direction.CAUSE, tuple(effect_indices), purviews=False, allow_neg=False)
            AA = pyphi.actual.account(transition,pyphi.Direction.CAUSE)


            alpha.append(CL.alpha)
            purview.append(CL.purview)
            motorstate.append(tuple(y[motor_indices]))
            account.append(AA)


            # avoiding recalculating the same occurrence twice
            tran.append(np.array(occurrence))
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



def pkl2df(path,experiment_list,n_trials=128,file_names=['version1_genome.pkl','version1_activity.pkl','version1_LOD_data.pkl'],
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
                animat.saveUniqueStates()
                #transitions, ids = animat.get_unique_transitions() # DOES NOT WORK NOW
                animat.saveUniqueTransitions()

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
                           'CC length' : ['TBD']})
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
                           'CC length' : ['TBD']})


                    df = df.append(df2)
    df2 = df.set_index(['Experiment','Run','Agent'])

    return df2


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



def backtrack_cause_brute_force(animat, trial, t, occurrence_ixs=None, max_backsteps=3, purview_type='union', debug=False):
    '''
    Brute force calculation of causal chain
    '''
    causal_chain = []

    backstep = 1
    end = False
    effect_ixs = occurrence_ixs
    cause_ixs = (0,1,2,3,4,5,6,)
    while not end and backstep <= max_backsteps and t>0:

        causes = get_actual_causes(animat, trial, t, cause_ixs, effect_ixs)
        n_causal_links = len(causes)

        if n_causal_links==0:
            end=True

        # use the union of the purview of all actual causes as the next occurrence (effect_ixs) in the backtracking
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

def get_hidden_ratio(animat,data):

    if type(data)==list:
        data = np.array(data)

    hidden_ratio = []
    print(data.shape)
    if len(data.shape) == 3:
        for t in range(len(data)):
            sensor_contribution = np.sum(data[t,:,animat.sensor_ixs])
            hidden_contribution = np.sum(data[t,:,animat.hidden_ixs])
            if not sensor_contribution + hidden_contribution == 0:
                hidden_ratio.append(hidden_contribution / (hidden_contribution + sensor_contribution))
            else:
                hidden_ratio.append(float('nan'))
    elif len(data.shape) == 2:
        for t in range(len(data)):
            sensor_contribution = np.sum(data[t,animat.sensor_ixs])
            hidden_contribution = np.sum(data[t,animat.hidden_ixs])
            if not sensor_contribution + hidden_contribution == 0:
                hidden_ratio.append(hidden_contribution / (hidden_contribution + sensor_contribution))
            else:
                hidden_ratio.append(float('nan'))
    else:
        print('something is wrong for calculating hidden ratio')

    return hidden_ratio


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
