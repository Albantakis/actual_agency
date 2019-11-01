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


### GENERAL FUNCTIONS USED BY ELSEWHERE

def state2num(state,convention='loli'):
    '''
    Function description
        Inputs:
            inputs:
        Outputs:
            outputs:
    '''
    # send in a state as a list
    num = 0
    if convention == 'loli':
        for i in range(len(state)):
            num += (2**i)*state[i]
    else:
        for i in range(len(state)):
            num += (2**i)*state[-(i+1)]

    # returns the number associated with the state
    return int(num)

def num2state(num,n_nodes,convention='loli'):
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
    if convention == 'loli':
        state.reverse()
        return state
    else:
        return state


### STRUCTURAL ANALYSIS

def number_of_connected_nodes(cm):
    return np.sum(np.sum(cm,0)*np.sum(cm,1)>0)

def number_of_connected_sensors(cm,n_sensors):
    return np.sum(np.sum(cm[:n_sensors,:],1)>0)

def number_of_connected_motors(cm,n_sensors,n_motors):
    return np.sum(np.sum(cm[:,n_sensors:n_sensors+n_motors],0)>0)

def number_of_densely_connected_nodes(cm,allow_self_loops=False):
    if not allow_self_loops:
        for i in range(len(cm)):
            cm[i,i] = 0
            return np.sum((np.sum(cm,0)*np.sum(cm,1))>0)

def number_of_sensor_hidden_connections(cm,n_sensors,n_motors):
    return np.sum(cm[:n_sensors,n_sensors+n_motors:]>0)

def number_of_sensor_motor_connections(cm,n_sensors,n_motors):
    return np.sum(cm[:n_sensors,n_sensors:n_sensors+n_motors]>0)

def number_of_hidden_hidden_connections(cm,n_sensors,n_motors):
    return np.sum(cm[n_sensors+n_motors:,n_sensors+n_motors:]>0)

def number_of_hidden_motor_connections(cm,n_sensors,n_motors):
    return np.sum(cm[n_sensors+n_motors:,n_sensors:n_sensors+n_motors]>0)


### DYNAMICAL ANALYSIS

def coalition_entropy(data):
    # get the set of all causes in the network
    potential_causes = tuple(tuple(cause) for cause in pt.get_states(8).tolist())
    # get the set of all actual causes
    all_causes = tuple(tuple(purview_elements) for trial in motor_cause for purview_elements in trial)
    # get the unique causes
    unique_data = list(set(all_causes))
    # count the number of occurrences for each unique cause
    occurrences = [all_causes.count(u) for u in unique_data]
    # calculate probability distribution (ordering does not matter)
    p = [occurrences[i]/len(all_causes) if i < len(unique_data) else 0 for i in range(len(all_causes))]
    # calculate the entropy
    H = -np.sum([0 if pp==0 else pp*np.log2(1/pp) for pp in p])
    return H

def LZ_algorithm(string):
    d={}
    w = ''
    i=1
    for c in string:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc]=wc
            w = c
        i+=1
    return len(d)


def LZ_complexity(data,dim='space',threshold=0,shuffles=10):

    # reshaping data to be 2d (assuming trial by time by node)
    shape = data.shape
    if len(shape)>2:
        data = data.reshape((np.prod(shape[:-1]),shape[-1]))
        print('Reshaping data, assuming time by node as last 2 dimensions')
    elif len(shape)==1:
        data = data.reshape((1,shape[0])).shape

    print(data.shape)

    # setting up variables for concatinating the data over time or space dimension
    if dim=='space':
        d1,d2=data.shape
    elif dim=='time':
        d2,d1=data.shape
    else:
        d1,d2=data.shape
        dim = 'space'
        print('Bad input. Concatinating in space')

    # making the concatenated (1D) string for calculating complexity
    s = ''
    for j in range(d2):
        for i in range(d1):
            if data[i,j]>threshold:
                s+='1'
            else:
                s+='0'

    # calculating raw LZ
    lz_raw = LZ_algorithm(s)

    # getting normalization
    randoms = []
    new_s = list(s)
    for i in range(shuffles):
        ran.shuffle(new_s)
        randoms.append(LZ_algorithm(new_s))

    return lz_raw/np.max(randoms)

from scipy.stats import entropy as ent
def effective_information(tpm,n_nodes):

    shp = tpm.shape
    # chacking that tpm is in state by state
    if not len(shp) == 2:
        tpm = pyphi.convert.to_2dimensional(tpm)
        tpm = pyphi.convert.state_by_node2state_by_state(tpm)
    if len(shp) == 2 and not shp[0] == shp[1]:
        tpm = pyphi.convert.state_by_node2state_by_state(tpm)

    determinism = np.mean([np.log2(2**n_nodes) - ent(tp) for tp in tpm]) # expected_entropy_of_out_weights
    degeneracy = np.log2(2**n_nodes) - ent(np.mean(tpm,0))

    return determinism - degeneracy

def predictive_information(data):
    '''
    data should be a state by state transition probability matrix
    '''
    def KL_divergence(p1,p2):
        KLD = []
        for i in range(len(p1)):
            if p1[i]==0 or p2[i]==0:
                KLD.append(0)
            else:
                KLD.append(p1[i]*np.log2(p1[i]/p2[i]))
        return KLD


    n_states = data.shape[0]
    p_V_given_v = np.array([(d/np.sum(d)).tolist() for d in data])
    p_V = np.sum(p_V_given_v,0)/np.sum(p_V_given_v)

    PI = np.sum(np.array([KL_divergence(p1,p_V) for p1 in p_V_given_v]))/n_states

    return PI


def entropy_of_cause_repertoires(data):
    # get the set of all causes in the network
    potential_causes = tuple(tuple(cause) for cause in pt.get_states(8).tolist())
    # get the set of all actual causes
    all_causes = tuple(tuple(purview_elements) for trial in motor_cause for purview_elements in trial)
    # get the unique causes
    unique_data = list(set(all_causes))
    # count the number of occurrences for each unique cause
    occurrences = [all_causes.count(u) for u in unique_data]
    # calculate probability distribution (ordering does not matter)
    p = [occurrences[i]/len(all_causes) if i < len(unique_data) else 0 for i in range(len(all_causes))]
    # calculate the entropy
    H = -np.sum([0 if pp==0 else pp*np.log2(1/pp) for pp in p])
    return H

### ACTTUAL CAUSATION ANALYSIS FUNCTIONS

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

def get_all_causal_links(animat):
    '''

    '''
    # calulate causal account for all unique transitions
    direct_causes = {}
    n_states = len(animat.unique_transitions)
    n = 0
    for t in animat.unique_transitions:
        #print('Finding causes for transition number {} out of {} unique transitions'.format(n,n_states))
        n+=1

        transition_number = state2num(list(t[0]+t[1]))
        cause_ixs = animat.sensor_ixs+animat.hidden_ixs+animat.motor_ixs
        effect_ixs = animat.sensor_ixs+animat.hidden_ixs+animat.motor_ixs
        Transition = pyphi.actual.Transition(animat.brain, t[0], t[1], cause_ixs, effect_ixs)
        CL = pyphi.actual.directed_account(Transition, pyphi.direction.Direction.CAUSE)
        direct_causes.update({transition_number : CL})
    return direct_causes


def get_union_of_causes(animat,transition,occurrence_ixs):
    '''

    '''

    full_cl = animat.causal_links[state2num(transition[0]+transition[1])].causal_links
    causes = ()
    for cl in full_cl:
        if set(cl.mechanism).issubset(occurrence_ixs):
            # adding cause purview to union of causes
            purview = cl.purview
            causes += purview

    return tuple(set(causes))


def backtrack_cause_enumerated(animat, trial, t, occurrence_ixs, max_backsteps=3, purview_type='union', debug=False):
    '''
    Calculation of causal chain using enumerated transitions
    '''
    causal_chain = np.zeros([max_backsteps,animat.n_nodes])

    backstep = 1
    end = False
    effect_ixs = occurrence_ixs
    while not end and backstep <= max_backsteps and t>=0 and not math.isnan(animat.enumerated_transitions[t]):
        # check if the effect_ixs have causes for current transition
        full_cl = animat.causal_links[animat.enumerated_transitions[t]].causal_links
        causes = ()
        for cl in full_cl:
            if set(cl.mechanism).issubset(effect_ixs):
                # adding cause purview to union of causes
                purview = cl.purview
                causes += purview
                causal_chain[max_backsteps-backstep,purview] += cl.alpha/len(purview)

        causes = tuple(set(causes))
        # checking if effect_ixs had any causes at all
        if len(causes)<1:
            end = True
        # or if all causes are sensors
        elif set(causes).issubset(animat.sensor_ixs):
            end = True
        else:
            backstep += 1
            t -= 1
            effect_ixs = causes

    return causal_chain.tolist()


def backtrack_cause(animat, trial, t, occurrence_ixs, max_backsteps=3, purview_type='union', debug=False):
    '''
    ### THIS IS ONLY HERE FOR BACKWARDS COMPATIBILITY, BUT IS KEPT HERE DUE TO
    ### NAME BEING INTUITIVE
    Function for tracking the causes of an occurrence back in time
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            t: the time of the second state in the transition (int)
            occurrence_ixs: the indices of the elements that constitute the occurrence under investigation
            max_backsteps: the maximum number of steps we track the causes back
            purview_type: name of the type of purview we use to track the causes
        Outputs:
            outputs: list of lists containing all cause purviews in the causal chain
    '''

    if not hasattr(animat,'node_labels'):
        ### the following is specially designed for the analysis of Juel et al 2019 and should be generalized
        if occurrence_ixs==None:
            occurrence_ixs = [2,3] if animat.n_nodes==8 else [1,2] # motor ixs
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
        if occurrence_ixs==None:
            occurrence_ixs = animat.motor_ixs

        if debug:
            print('MAKE A PROPER label_dict FOR RUNNING DEBUG')

    if not hasattr(animat, 'enumerated_transitions'):
        causal_chain = backtrack_cause_brute_force(animat, trial, t, occurrence_ixs, max_backsteps, purview_type, debug)

        # else, if causal links for all unique transitions have been calculated
    else:
        causal_chain = backtrack_cause_enumerated(animat, trial, t, occurrence_ixs, max_backsteps, purview_type, debug)

    return causal_chain

def get_average_causal_chain(animat,max_backsteps=3):
    '''
    Calculation of average causal chain leading to motor occurrences
    '''
    n_trials = animat.brain_activity.shape[0]
    n_steps = animat.brain_activity.shape[1]
    n_nodes = animat.brain_activity.shape[2]
    CC = np.zeros((n_trials,max_backsteps,n_nodes))
    for tr in range(n_trials):
        for t in range(max_backsteps,n_steps):
            CC[tr,:,:] = backtrack_cause(animat, tr, t, animat.motor_ixs, max_backsteps)

    return np.mean(CC,0)


def backtrack_cause_trial(animat,trial,max_backsteps=3,occurrence_ixs=None,purview_type='union'):
    '''
    Calculates the causal chain leading to an occurrence for all transitions in a trial
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            max_backsteps: the maximum number of steps we track the causes back (int). also the first timestep tracked back
            occurrence_ixs: the indices of the elements that constitute the occurrence under investigation
            purview_type: name of the type of purview we use to track the causes
        Outputs:
            causal_chain: a list of backtracking patterns for each timestep in a trial
    '''

    causal_chain = []
    n_times = animat.brain_activity.shape[1]
    '''
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
    '''
    if occurrence_ixs is None:
        occurrence_ixs = animat.motor_ixs

    for t in range(max_backsteps,n_times):
        state = animat.brain_activity[trial][t]

        if not -1 in state:
            if occurrence_ixs == 'MC':
                ix = get_complex_indices(animat,tuple(state))
            else:
                ix = occurrence_ixs
            CC = backtrack_cause(animat, trial, t, ix, max_backsteps, purview_type)
            causal_chain.append(CC)

    return causal_chain

def backtrack_all_activity(animat,max_backsteps=3,occurrence_ixs=None,purview_type='union'):
    '''
    Calculates the causal chain leading to an all motor occurrences for an animat
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            max_backsteps: the maximum number of steps we track the causes back (int). also the first timestep tracked back
            occurrence_ixs: the indices of the elements that constitute the occurrence under investigation
            purview_type: name of the type of purview we use to track the causes
        Outputs:
            causal_chain: a list of backtracking patterns for each timestep in a trial
    '''

    causal_chain = []
    n_times = animat.brain_activity.shape[1]
    n_trials = animat.brain_activity.shape[0]

    if occurrence_ixs is None:
        occurrence_ixs = animat.motor_ixs
    print('Calculating causal chain for a full set of brain activity')
    aux = ran.rand()
    if aux<0.02:
        print('Have patience young padawan!')
    elif aux<0.04:
        print('have faith! It will finish eventually...')
    elif aux<0.05:
        print("this is a chicken, for your entertainment      (  ')>  ")
    elif aux<0.06:
        print('This might be a good time for a coffee')
    for tr in range(n_trials):
        cc_tr = []
        for t in range(max_backsteps,n_times):
            CC = backtrack_cause(animat, tr, t, occurrence_ixs, max_backsteps, purview_type)
            cc_tr.append(list(np.mean(CC,0)))
        causal_chain.append(cc_tr)
    return causal_chain


def get_alpha_distribution(animat,t,effect_ixs):

    alpha_distribution = np.zeros((animat.n_nodes))
    if not math.isnan(animat.enumerated_transitions[t]):
        full_cl = animat.causal_links[animat.enumerated_transitions[t]].causal_links
        causes = ()
        for cl in full_cl:
            if set(cl.mechanism).issubset(effect_ixs):
                # adding cause purview to union of causes
                purview = cl.purview
                for p in purview:
                    alpha_distribution[p] += cl.alpha/len(purview)

    return alpha_distribution.tolist()


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


def get_causal_history(animat, trial, occurrence_ixs=None,MC=False):
    '''
    ### SIMILAR TO OLD calc_causal_history(), BUT BETTER SUITED TO NEWER VERSION
    Calculates animat's direct cause history, defined as the direct causes of
    every transition (only motor or not) across a trial.
        Inputs:
            animat: object where the animat brain and activity is defined
            trial: the trial number under investigation (int)
            only_motor: indicates whether the occurrence under investigation is only motors or the wholde network
        Outputs:
            direct_cause_history: list of lists of irreducible cause purviews
    '''

    direct_cause_history = []
    direct_alpha_history = []
    dims = animat.brain_activity.shape
    n_times = dims[1]
    if len(dims)==2:
        start_step = trial*(n_times - 1) +1
        trial = 0
        print('data_dimension might be wrong and lead to errors. Get more than one trial!')
    elif len(dims)==3:
        start_step = 1 # index of first transition in given trial

    # if no particular occurrence indices are given. Motors are assumed,
    # unless 'complex' is True, in which case the occurrence is always the complex in current state
    if not MC:
        occurrence_ixs = animat.motor_ixs if occurrence_ixs==None else occurrence_ixs

        for t in range(start_step,start_step + n_times-1):

            transition = animat.get_transition(trial, t, trim=False)
            if not -1 in transition[0] and not -1 in transition[1]:
                if not MC == None:
                    cause_ixs = get_union_of_causes(animat,transition,occurrence_ixs)
                    direct_cause_history.append([int(1) if i in cause_ixs else int(0) for i in range(animat.n_nodes)])
                    direct_alpha_history.append(get_alpha_distribution(animat,t-1,occurrence_ixs))
                else:
                    direct_cause_history.append(np.zeros((animat.n_nodes)).astype(int).tolist())
                    direct_alpha_history.append(np.zeros((animat.n_nodes)).tolist())
        return direct_cause_history,direct_alpha_history

    # calulcating history of causes of the state of the complex
    else:
        for t in range(start_step,start_step + n_times-1):
            if not math.isnan(animat.enumerated_states[trial*n_times + t]):
                MCs = animat.MCs[animat.enumerated_states[trial*n_times + t]]
                if not MCs == None:
                    occurrence_ixs = MCs.subsystem.node_indices
                    transition = animat.get_transition(trial, t, trim=False)
                    cause_ixs = get_union_of_causes(animat,transition,occurrence_ixs)
                    direct_cause_history.append([int(1) if i in cause_ixs else int(0) for i in range(animat.n_nodes)])
                    direct_alpha_history.append(get_alpha_distribution(animat,t-1,occurrence_ixs))
                else:
                    direct_cause_history.append(np.zeros((animat.n_nodes)).astype(int).tolist())
                    direct_alpha_history.append(np.zeros((animat.n_nodes)).tolist())
            else:
                return direct_cause_history,direct_alpha_history
        return direct_cause_history,direct_alpha_history



### IIT related FUNCTIONS

def get_complex_indices(animat,state):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist
    if type(state)==int:
        if animat.MCs[state] == None:
            return ()
        else:
            return animat.MCs[state].subsystem.node_indices
    else:
        state_num = state2num(state)
        if animat.MCs[state_num] == None:
            return ()
        else:
            return animat.MCs[state_num].subsystem.node_indices


def get_phi_from_subsystem(animat, state, node_indices = (4,5,6,7)):

    #calculate phi on specified subsystem,
    #4,5,6,7 all hidden nodes of the animat
    subsystem = pyphi.Subsystem(animat.brain, state, node_indices)
    phi = pyphi.compute.phi(subsystem)

    return phi

def get_phi_from_complexes(animat, state):
    complexes = pyphi.compute.network.complexes(animat.brain, state)
    if len(complexes) == 0:
        phi = 0.0
    elif len(complexes) > 1:
        phi = []
        for complex_num in range(len(complexes)):
                phi.append(complexes[complex_num].phi)
    else:
        phi_table[run].append(complexes[0].phi)
    return phi

def system_irreducibility_analysis(animat):
    '''

        Inputs:

        Outputs:
    '''

    sias = {}
    for s in animat.unique_states:
        sia = pyphi.compute.complexes(animat.brain,s)
        if len(sia)==0:
            sias.update({state2num(s) : None})
        else:
            sias.update({state2num(s) : sia})
    animat.sias = sias


def major_complex(animat):
    '''

        Inputs:

        Outputs:
    '''

    MCs = {}
    n_states = len(animat.unique_states)
    n = 0
    for s in animat.unique_states:
        #print('Finding MC for state number {} out of {} unique states'.format(n,n_states))
        n+=1

        MC = pyphi.compute.major_complex(animat.brain,s)
        if MC.phi==0:
            MCs.update({state2num(s) : None})
        else:
            MCs.update({state2num(s) : MC})
    animat.MCs = MCs



def phi_from_MCs(animat):
    '''

        Inputs:

        Outputs:
    '''

    phis = []
    for s in animat.unique_states:

        if animat.MCs[state2num(s)] == None:
            phis.append(0)
        else:
            phis.append(animat.MCs[state2num(s)].phi)
    animat.phis = phis


def get_complex_from_past_state_in_transtition(animat,transition):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist

    # Find the past state in the transition and pick out the complex of that state
    past_state = transition[0]
    past_state_num = state2num(past_state)
    MC = animat.MCs[past_state_num]

    MC_nodes = () if MC == None else MC.subsystem.node_indices

    return MC_nodes


def get_complex_from_current_state_in_transtition(animat,transition):
    '''
        Inputs:
        Outputs:
    '''
    current_state = transition[1]
    current_state_num = state2num(current_state)
    MC = animat.MCs[current_state_num]

    MC_nodes = () if MC == None else MC.subsystem.node_indices

    return MC_nodes


def get_complex_purview_overlap(animat,transition,occurrence_ixs):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist
    if not occurrence_ixs == 'MC':
        # find cause ixs
        causes = get_union_of_causes(animat,transition,occurrence_ixs)

        # find MC from the past state
        MC = get_complex_from_past_state_in_transtition(animat,transition)

        # return % overlap
        if causes == () or MC == ():
            return 0
        else:
            intersection = set(causes).intersection(MC)
            union = set(causes).union(MC)
            pct_overlap = len(intersection)/len(union)

            return pct_overlap
    else:
        # find cause ixs
        MC_curr = get_complex_from_current_state_in_transtition(animat,transition)
        if not MC_curr == ():
            causes = get_union_of_causes(animat,transition,MC_curr)
        else:
            causes = ()

        # find MC from the past state
        MC = get_complex_from_past_state_in_transtition(animat,transition)

        # return % overlap
        if causes == () or MC == ():
            return 0
        else:
            intersection = set(causes).intersection(MC)
            union = set(causes).union(MC)
            pct_overlap = len(intersection)/len(union)

            return pct_overlap



def stability_of_complex_over_transition(animat,transition):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist

    # Find complex in each state of the transition
    prev_state_MC_nodes = get_complex_from_past_state_in_transtition(animat,transition)
    curr_state_MC_nodes = get_complex_from_current_state_in_transtition(animat,transition)

    # Find overlap between complexes in the two states
    if prev_state_MC_nodes == () or curr_state_MC_nodes == ():
        return 0
    else:
        intersection = set(prev_state_MC_nodes).intersection(curr_state_MC_nodes)
        union = set(prev_state_MC_nodes).union(curr_state_MC_nodes)
        pct_overlap = len(intersection)/len(union)

        return pct_overlap

def history_of_complexes(animat,trial=None,only_state_changes=True):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist

    # reshaping the enumerated states in case it not organized as trials
    n_trials = animat.n_trials
    n_steps = animat.n_timesteps
    state_change = [True]

    # getting the full history of complexes
    state_nums = [[animat.enumerated_states[t*n_steps+s] for s in range(n_steps)] for t in range(n_trials)]
    complex_indices = [[get_complex_indices(animat,state_num) for state_num in trial if not math.isnan(state_num)] for trial in state_nums]
    complex_history = [[[1 if i in MC else 0 for i in range(animat.n_nodes)] for MC in trial] for trial in complex_indices]

    if only_state_changes:
        # only including state transitions where state actually changed
        true_history = []
        for t in range(len(state_nums)):
            true_history_trial = [complex_history[t][0]]
            for s in range(1,len(state_nums[0])):
                if not state_nums[t][s] == state_nums[t][s-1]:
                    true_history_trial.append(complex_history[t][s])
            true_history.append(true_history_trial)

        return true_history if trial==None else true_history[trial]
    else:
        return complex_history if trial==None else complex_history[trial]


def actual_cause_of_complex(animat,trial=None,step=None):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist

    #

    return







### DATA ANALYSIS FUNCTIONS
'''
NEXT TWO SCRIPTS (TO CALCULATE LZ) ARE ADAPTED FROM SCHARTNER ET AL 2015.
MAIN DIFFERENCE IS THAT THEY TAKE A BINARY LIST OF INPUTS RATHER THAN TIMESERIES
AND THAT THERE ARE NOW THREE LETTERS IN THE ALPHABET (TO ALLOW FOR INHIBITION)
'''

def cpr(string):
    '''
    Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
    '''
    d={}
    w = ''
    i=1
    for c in string:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc]=wc
            w = c
            i+=1
    return len(d)

def LZc(X):
    '''
    Compute LZc and use shuffled result as normalization
    '''
    # making string out of list
    s=''
    for j in X:
        if j==1:
            s+='1'
        elif j==2:
            s+='2'
        else:
            s+='0'

    np.random.shuffle(X)
    w=''
    for j in X:
        if j==1:
            w+='1'
        elif j==2:
            w+='2'
        else:
            w+='0'
    return cpr(s)/float(cpr(w))


def calculate_PCI(animat,perturb_idx,repetitions,steps,st_devs=1.96,concat_PCI='time',nodes=None):
    '''
        Inputs:
        Outputs:
    '''

    # Check if necessary animat properties exist

    # getting some parameters
    n_nodes = animat.n_nodes
    tpm = animat.brain.tpm
    tpm2d = pyphi.convert.to_2dimensional(animat.brain.tpm)
    n_states = tpm2d.shape[0]
    perturbations = repetitions*n_states
    expected_activity = np.mean(tpm2d,0)
    perturb_state = np.ones(len(perturb_idx))

    # getting random numbers for comparisons to decide state update
    thresholds = np.random.random((perturbations, steps, n_nodes))
    all_responses = []

    # running stimulations
    for s in range(perturbations):
        # previous state
        state = num2state(s%n_states,n_nodes)
        # perturbing
        state = [state[i] if i not in perturb_idx else 1 for i in range(n_nodes)]
        # find the response to perturbations
        response = []
        for t in range(steps):
            # get next state probabilities
            next_state_p = tpm[tuple(state)]
            # comparing with threshold to get next state
            next_state = [1 if p>t else 0 for p,t in zip(next_state_p,thresholds[s,t,:])]
            # adding next_state to response, and updating current state
            response.append(next_state)
            state = copy.deepcopy(next_state)

        all_responses.append(response)

    # getting stats for each node
    res = np.array(all_responses)
    mean_response = np.zeros((n_nodes,steps))
    sem = np.zeros((n_nodes,steps))
    for n in range(n_nodes):
        mean_response[n,:], sem[n,:] = get_bootstrap_stats(res[:,:,n],n=500)

    # binarizing response
    # first generating matrices for high and low thresholds for significant
    # (de)activation, based on expected activation and standard error of the means
    t_high = np.zeros((n_nodes,steps))
    t_low = np.zeros((n_nodes,steps))
    binary_response = np.zeros((n_nodes,steps))
    for n in range(n_nodes):
        for s in range(steps):
            t_high[n,s] = expected_activity[n] + sem[n,s]*st_devs
            t_low[n,s] = expected_activity[n] - sem[n,s]*st_devs

            if mean_response[n,s] > t_high[n,s]:
                binary_response[n,s] = 1
            elif mean_response[n,s] < t_low[n,s]:
                binary_response[n,s] = 2

    # Calculate PCI
    if not nodes == None:
        binary_response = binary_response[nodes,:]

    concat_data = []
    if concat_PCI == 'time':
        for data in binary_response.astype(int):
            concat_data.extend(data)
    else:
        for data in np.transpose(binary_response.astype(int)):
            concat_data.extend(data)

    PCI = LZc(concat_data)

    return PCI, all_responses, mean_response, binary_response, expected_activity

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


def get_unique_states_binary(activity):
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
