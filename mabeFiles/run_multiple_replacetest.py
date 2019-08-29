from run_experiment import createFolder, run_experiment
import subprocess as sp
import pandas as pd
import numpy as np
import pickle
from matplotlib import pyplot as plt
import os
import fileinput

# the path to your MABE installation (this file should be in the same folder)
#mabe_path = '/Users/bjornjuel/projects/mabe_development/mabe/'
mabe_path = os.path.dirname(os.path.realpath(__file__))+'/'

# name of folder where basic settings and experiments are stored
name_folder = 'Experiments/'

# the name of your experiment
name_experiment = 'replacetest/'

# the names of your cfg files defining the changes to the standard settings writeSnapshotDataFiles
name_exp1 = 'experiment_diagnostic.cfg'
name_exp2 = 'experiment_record_activity_diagnostic.cfg'

# make a list of all the experiments you want to run (must match the houtput files defined later)
cmd = [name_folder+name_experiment+name_exp1, name_folder+name_experiment+name_exp2]

# define prefixes you want for your output files
fileprefix_exp1 = 'exp_'
fileprefix_exp2 = 'act_'
filename_activityfile = 'markov_IO_map.csv'

# define the names of the date files produced by experiments
datafile = mabe_path+name_folder+name_experiment+fileprefix_exp1+'LOD_data.csv'
genomefile = mabe_path+name_folder+name_experiment+fileprefix_exp1+'LOD_organisms.csv'
activityfile = mabe_path+name_folder+name_experiment+filename_activityfile
data = []
genome = []
activity = []

# changing directory to the mabe path
os.chdir(mabe_path)

# now we update the cfg files with the paths and names defined above
# This must be done for each experiment cfg file
# first main experiment (individual for statement is needed for each change)
for line in fileinput.input(name_folder+name_experiment+name_exp1, inplace=True):
    print(line.replace('path = DO NOT CHANGE', 'path = '+mabe_path), end='')
for line in fileinput.input(name_folder+name_experiment+name_exp1, inplace=True):
    print(line.replace('outputPrefix = DO NOT CHANGE', 'outputPrefix = '+name_folder+name_experiment), end='')
for line in fileinput.input(name_folder+name_experiment+name_exp1, inplace=True):
    print(line.replace('filePrefix = DO NOT CHANGE', 'filePrefix = '+fileprefix_exp1), end='')

# second experiment (here, recording activity)
for line in fileinput.input(name_folder+name_experiment+name_exp2, inplace=True):
    print(line.replace('path = DO NOT CHANGE', 'path = '+mabe_path), end='')
for line in fileinput.input(name_folder+name_experiment+name_exp2, inplace=True):
    print(line.replace('outputPrefix = DO NOT CHANGE', 'outputPrefix = '+name_folder+name_experiment), end='')
for line in fileinput.input(name_folder+name_experiment+name_exp2, inplace=True):
    print(line.replace('filePrefix = DO NOT CHANGE', 'filePrefix = '+fileprefix_exp2), end='')
for line in fileinput.input(name_folder+name_experiment+name_exp2, inplace=True):
    print(line.replace('recordIOMap_fileName = DO NOT CHANGE', 'recordIOMap_fileName = '+filename_activityfile), end='')

# then we run the experiments
runs = 5

for r in list(range(0,runs)):
    print(['run number ' + str(r)])
    # Running experiment
    run_experiment(cmd[0])
    # Getting data from file
    data.append(pd.read_csv(datafile))
    genome.append(pd.read_csv(genomefile))
    # Running recorders
    run_experiment(cmd[1])
    # Getting recordings from file
    activity.append(pd.read_csv(activityfile))

    with open(name_folder+name_experiment+'LOD_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open(name_folder+name_experiment+'genome.pkl', 'wb') as f:
        pickle.dump(genome, f)

    with open(name_folder+name_experiment+'activity.pkl', 'wb') as f:
        pickle.dump(activity, f)
