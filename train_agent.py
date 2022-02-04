from __future__ import division
import argparse
from ConfigParser import SafeConfigParser
from PIL import Image
import numpy as np
import gym
from gym import wrappers
import pyglet
from pyglet.gl import *
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Input, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
import keras.backend as K
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, TensorBoard
import scipy.io as sio
import os
import sys
os.chdir('..')
from os import listdir
from os.path import isfile, join, expanduser
sys.path.insert(0, os.path.join(expanduser("~"),'keras-rl/TAMER'))
sys.path.insert(0,'../')
from keras.regularizers import l1l2
from auto_encoder import SimpleAutoEncoder
from rl.agents.TAMER import TAMERAgent, TAMERAgentReplay, TAMERAgentCreditAssignment
from rl.agents.tetris_agent import Super_Vanilla_TAMERAgent, Deep_TAMERAgent
import tensorflow as tf


if __name__ == '__main__':        
        
    #Command-Line Input Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file',type=str,default='ExperimentScripts/ConfigTAMER_Experiment.ini')
    parser.add_argument('--config',type=str,default='generic')
    parser.add_argument('--practice',type=bool,default=False)
    parser.add_argument('--delay',type=float,default=0.0)
    
    args = parser.parse_args()
    
    #Configurtion File Input Arguments
    cf_parser= SafeConfigParser()
    cf_parser.read(args.config_file)
    
    
    args.description = cf_parser.get(args.config,'description')
    args.folder = cf_parser.get(args.config,'folder')    
    args.sleep_interval = args.delay
    
    args.description = args.description + '_vanilla_'
    
    if args.practice == True:
        args.description = 'practice'
        args.folder='TAMER_Experiments_aaai18/practice/'
    
    
    # Get the environment and extract the number of actions.
    env_name = 'Tetris-v0'
    env = gym.make(env_name)
    np.random.seed(123)
    env.seed(123)
    
    
    # Next, we build our TAMER model - This does not need to be done here as one is already defined within tamer
    # but we do it here if we want to change it (which we wouldn't since it's vanilla)

    # vanilla tamer 46 --> 1  linear model
    input_shape = 46  # 46 features
    state_input = Input(shape = (input_shape,))
    pred_reward = Dense(1, init = 'zero')(state_input)
    TAMERmodel = Model(input = state_input, output = pred_reward)
    TAMERmodel.summary()
    
    
    # Create and compile our TAMER agent
    tamer = Super_Vanilla_TAMERAgent(model = TAMERmodel, env = env, sleep_interval = args.sleep_interval,
                                     save_file_name = '{}{}_{}'.format(args.folder, args.description, env_name))
    tamer.compile(optimizer = SGD(lr=0.000005 / 47.0))
    
    weights_filename = '{}{}_{}_weights.h5f'.format(args.folder,args.description,env_name)
    checkpoint_weights_filename = args.folder+args.description + '_' + env_name + '_weights_{step}.h5f'
    log_filename = '{}{}_{}_log.json'.format(args.folder,args.description,env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=50)]        
    callbacks += [TensorBoard(log_dir='{}{}_{}_tensorboard_data'.format(args.folder,args.description,env_name), agent = tamer)]               
                              
    # train tamer agent
    tamer.fit(env, callbacks=callbacks, visualize=True)
   
