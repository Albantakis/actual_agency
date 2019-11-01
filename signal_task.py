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
import math

import pyphi
import pyanimats as pa
import pyTPM as pt
import analysis as agency


'''
Data analysis functions.

'''
def getBrainActivity(data, n_agents=1, n_trials=64, n_nodes=8, n_sensors=2,n_hidden=4,n_motors=2, world_height = 200):
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
    n_transitions = world_height
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
                    sensor = np.ones(n_sensors)*-1
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                else:
                    sensor = np.fromstring(str(data['input_LIST'][ix]), dtype=int, sep=',')[:n_sensors]
                    hidden = np.fromstring(data['hidden_LIST'][ix-1], dtype=int, sep=',')
                    motor = np.fromstring(data['output_LIST'][ix-1], dtype=int, sep=',')
                nodes = np.r_[sensor, motor, hidden]
                brain_activity[a,i,j,:] = nodes
    return brain_activity


def get_genome(genomes, run, agent):
    genome = genomes[run]['GENOME_root::_sites'][agent]
    genome = np.squeeze(np.array(np.matrix(genome)))
    return genome


def parse_task_and_actions(animat):

# for the actual agency task

    activity = animat.brain_activity

    # check dimensionality of activity
    if len(activity.shape) == 3:
        trials,times,nodes = activity.shape
    elif len(activity.shape) == 2:
        trials = 1
        times,nodes = activity.shape
        activity = numpy.reshape(activity, activity.shape + (1,))
    else:
        print('check the dimensions of your data array')
        return None

    prev_task = (-1,-1)
    action = False
    thinking = (0,0)

    task_list = []
    action_list = []

    for trial in range(trials):
        task_list_trial = []
        action_list_trial = []
        for t in range(times-1):
            new_task = tuple(activity[trial,t,animat.sensor_ixs])
            motor_state = tuple(activity[trial,t,animat.motor_ixs])

            if not new_task == (-1,-1):
                if not new_task == prev_task:
                    task_list_trial.append(new_task+(0,t,))
                    if not motor_state == thinking:
                        action_list_trial.append(motor_state+(t,))
                        action = True

                elif new_task == prev_task and not motor_state == thinking:
                    task_list_trial.append(new_task+(1,t,))
                    action_list_trial.append(motor_state+(t,))
                    action = True

                elif new_task == prev_task and motor_state == thinking:
                    action = False

                prev_task = new_task

        task_list.append(task_list_trial)
        action_list.append(action_list_trial)

    animat.task_list = task_list
    animat.action_list = action_list

    return task_list, action_list







'''
Data management functions.
'''





'''
Plotting functions.
'''
