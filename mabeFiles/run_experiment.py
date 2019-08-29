import subprocess
import pandas as pd
import pickle
import numpy as np
from configparser import ConfigParser
import os
import sys

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def run_experiment(experiment_file):
    print(experiment_file)

    if not os.path.isfile(experiment_file):
        raise FileNotFoundError('{} does not exist.'.format(experiment_file))
        
    parser = ConfigParser()
    parser.optionxform = str # Case sensitive variables
    parser.read(experiment_file)

    path = (parser.get('shell', 'path'))
    executeable = (parser.get('shell', 'exec'))
    args = (parser.get('shell', 'args'))
    createFolder(parser.get('GLOBAL', 'outputPrefix'))

    plf = (parser.get('population_loader','initPop'))
    with open(os.path.join(path, "population_loader.plf"),"w") as f:
        f.write(plf)

    args2 = "-p"
    sections = parser.sections()[2:]
    for section in sections:
        variables = list(parser[section])
        args2 += ' '
        args2 += ' '.join(['{}-{} {}'.format(section,var,parser[section][var]) for var in variables])

    cmd = "./" + executeable + " " + args + " " + args2
    print (cmd)
    os.chdir(path)
    print(os.getcwd())
    subprocess.call(cmd, shell=True,stderr=True)

if __name__ == '__main__':
    if len(sys.argv)==1:
        raise TypeError("experiment_file not specified.")
    else:
        run_experiment(sys.argv[1])
