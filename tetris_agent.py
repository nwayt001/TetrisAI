#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 19:40:15 2017

@author: nicholas

TAMER agent for playing tetris
"""
import gym
import copy
import time
from rl.callbacks import TestLogger, TrainEpisodeLogger, TrainIntervalLogger, Visualizer, CallbackList
from copy import deepcopy
import numpy as np
import keras.backend as K
from keras.layers import Lambda, Input, merge, Layer, Dense, Convolution2D, Flatten,BatchNormalization, MaxPooling2D, AveragePooling2D
from keras.models import Model
from keras.activations import relu,linear
from rl.input_methods import HumanInput
from rl.memory import RingBuffer, zeroed_observation
import tensorflow as tf
import random
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import pickle
from auto_encoder import SimpleAutoEncoder
from rl.callbacks import TensorBoard

class TetrisAgent(object):
    def __init__(self, env, input_method = 'keyboard', nb_state_frames = 1, save_file_name = 'tamer',
                sleep_interval =0.1, processor = None):
        self.tetris_env = []
        self.env = env
        self.input_method = input_method
        self.nb_state_frames = nb_state_frames    
        self.tempState = gym.make('Tetris-v0')
        self.worldState = []
        self.step = 0
        self.processor = processor
        self.sleep_interval = sleep_interval        
        self.human_input = HumanInput.HumanInput('Generic',default_action = 0, input_method = self.input_method)
        self.save_file_name = save_file_name
        self.run_tensorboard_step_metrics = True
        self.nb_feedback_episode = 0
        self.nb_feedback_total = 0
        self.human_reward = 0
        
    # method for determining duplicate tetris pice positions for tree pruning
    def isInExtActList(self, state, moves):
        inList = False
        for move in moves:
            if state.stone == move.stone and state.stone_x == move.currentX and state.stone_y == move.currentY:
                inList = True
        return inList

    # custom object for representing a final tetris position/action
    class ExtendedTetrisAction(object):
        def __init__(self, currentRotation, currentX, currentY, blockMobile, stone, actList, board):
            self.currentRotation = currentRotation
            self.currentY = currentY
            self.currentX = currentX
            self.blockMobile = blockMobile
            self.stone = stone
            self.actList = actList
            self.state_features = []
            self.state_state_features = []
            self.state_value = []
            self.board = board
            
        def setState(self, tetrisEnv):
            tetrisEnv.currentRotation = self.currentRotation
            tetrisEnv.blockMobile = self.blockMobile
            tetrisEnv.stone_x = self.currentX
            tetrisEnv.stone_y = self.currentY
            tetrisEnv.stone = self.stone
            
        def addNoneAction(self):
            self.actList.append(4)
    
    def setStateFromPos(self, position):
        self.tempState.reset()
        self.tempState.board = position.board
        self.tempState.stone = position.stone
        self.tempState.stone_x = position.currentX
        self.tempState.stone_y = position.currentY
    
    # tree search method for determining all final pice placements
    def getAllPiecePos(self, worldState):
        #tempState = copy.copy(worldState)
        
        actOrder = [4, 0, 1, 2, 3]
        extendedActList = []
        
        # make copy of world state
        
        self.tempState.reset()
        self.tempState.board = worldState.board
        self.tempState.stone = worldState.stone
        self.tempState.stone_x = worldState.stone_x
        self.tempState.stone_y = worldState.stone_y
        self.tempState.currentStoneId = worldState.currentStoneId
        self.tempState.currentRotation = worldState.currentRotation
        self.tempState.blockMobile = worldState.blockMobile
        self.tempState.nextStoneId = worldState.nextStoneId
        self.tempState.next_stone = worldState.next_stone
        
        # start move
        startMove = self.ExtendedTetrisAction(self.tempState.currentRotation, self.tempState.stone_x, self.tempState.stone_y, self.tempState.blockMobile, self.tempState.stone, [], self.tempState.board)
        liveMoves = []
        liveMoves.append(startMove)
        
        # search tree of possible moves, pruning duplicates
        while len(liveMoves) > 0:
            #print(' len liveMoves: '+str(len(liveMoves)))
            nextLevelLiveMoves = []
            for actNum in actOrder: # make each possible move
                for liveMove in liveMoves:
                    liveMove.setState(self.tempState)
                    if(not self.tempState.take_action(actNum)):
                        #print('illegal action')
                        continue
                    #print(" blockMobile before "+  str(tempState.blockMobile))
                    board = self.tempState.update()
                    
                    # if move is duplicate, continue
                    if self.isInExtActList(self.tempState, extendedActList):
                        #print('duplicate in extendedActList')
                        continue
                    
                    if (self.isInExtActList(self.tempState, nextLevelLiveMoves) or self.isInExtActList(self.tempState, liveMoves)) and self.tempState.blockMobile:
                        #print('duplicate in nextLevelLiveMoves or liveMoves')
                        continue
                    
                    # else, make new extended action
                    thisActList = copy.deepcopy(liveMove.actList)
                    thisActList.append(actNum)
                    
                    thisMove = self.ExtendedTetrisAction(self.tempState.currentRotation, self.tempState.stone_x, self.tempState.stone_y, self.tempState.blockMobile, self.tempState.stone, thisActList, board)
                    #print(" blockMobile after "+  str(tempState.blockMobile))
                    # if block is active, make extendedTetrisAction and add to liveMoves
                    if self.tempState.blockMobile:
                        nextLevelLiveMoves.append(thisMove)
                        #print('len nextLevelLiveMoves '+str(len(nextLevelLiveMoves)))
                    else:
                        extendedActList.append(thisMove)
                        
            liveMoves = nextLevelLiveMoves
            
        return extendedActList
           
            
    def visualizePiecePositions(self, worldState, extendedActList):
        # test to visualize
        #tempState = copy.copy(worldState)
        for actionset in extendedActList:   
            self.tempState.reset()
            self.tempState.board = copy.deepcopy(worldState.board)
            self.tempState.stone = worldState.stone
            self.tempState.stone_x = worldState.stone_x
            self.tempState.stone_y = worldState.stone_y
            self.tempState.currentStoneId = worldState.currentStoneId
            self.tempState.currentRotation = worldState.currentRotation
            self.tempState.blockMobile = worldState.blockMobile
            self.tempState.nextStoneId = worldState.nextStoneId
            self.tempState.next_stone = worldState.next_stone
            for action in actionset.actList:
                self.tempState.take_action(action)
                self.tempState.update_and_draw()
                #tempState.step(action)
                self.tempState.render()


class Super_Vanilla_TAMERAgent(TetrisAgent):
    def __init__(self, model=None, *args, **kwargs):
        super(Super_Vanilla_TAMERAgent, self).__init__(*args, **kwargs)
        self.model = model
        
        
    def compile(self, optimizer = None, learning_rate = 0.001, metrics=[]):
        # construct generic model if one is not specified
        if self.model == None:
            input_shape = 46
            state_input = Input(shape = (input_shape,))
            
            pred_reward = Dense(1, init = 'zero')(state_input)
            
            self.model = Model(input = state_input, output = pred_reward)
            #self.model.summary()
        
        # compile the model
        if optimizer == None:
            self.model.compile(optimizer = SGD(lr=0.000005 / 47.0), loss='mse')
        else:
            self.model.compile(optimizer = optimizer, loss='mse')
        
        self.compiled = True
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)
        
    def save_and_exit(self, training):
        if training:
            # save current model weights
            self.model.save_weights(self.save_file_name + '_weights.hf5')
        
        # exit
        exit(0)
            
    def fit(self, nb_steps=100000, callbacks = None, verbose = 4, visualize = True, 
            training = True, model_file_name = 'Vanilla_Tetris-v0-100-lines.hf5', max_episode_steps = 500):

        if not training:
            # load pre-trained model weights        
            self.load_weights(model_file_name)
        else:
            if not self.compiled:
                raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        
        callbacks = [] if not callbacks else callbacks[:]
        self.callbacks = callbacks    
        self.callbacks = CallbackList(callbacks)
        if hasattr(self.callbacks, 'set_model'):
            self.callbacks.set_model(self)
        else:
            self.callbacks._set_model(self)
        self.callbacks._set_env(self.env)
        
        self.nb_steps = nb_steps
        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        done = True
        episode_scores = []
        while episode < self.nb_steps:
            
            if done:
                # save episode metrics after each episode (unless we are just starting)
                episode_reward = self.env.lines
                if not training:
                    episode_scores.append(episode_reward)
                if episode_step is not None:
                    episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            }
                    self.callbacks.on_episode_end(episode, episode_logs)
                print('Total Steps: {}, Episode Steps: {}, Total Episodes: {},  Lines Cleared: {}'.format(self.step,episode_step,episode,episode_reward))  
                episode_step = 0
                episode_reward = 0.
                episode += 1
                # Obtain the initial observation by resetting the environment.
                self.env.reset()
                if visualize:
                    self.env.render()
                
                self.prev_action = None
                self.prev_state_state_feats = None
            
            # for current piece, get all possible piece placements
            piecePlacements = self.getAllPiecePos(self.env)
            
            # for current state, extract features
            state0_feat = self.env.getFeatures(self.env.board)
            
            # for each piece placement, extract features
            state1_feats = []
            for piecePlacement in piecePlacements:
                state1_feats.append(self.env.getFeatures(piecePlacement.board))
                
            # get state_state features
            state_state_feats = []
            for state1_feat in state1_feats:
                state_state_feats.append(state1_feat - state0_feat)
                
            # use tamer model to get best action
            action = self.forward(state_state_feats)
            
            # with final piece placement, step through all actions from start to finish
            # with some time delay, while also checking for human reward in that time
            # period to use as the label for the previous time block
            atomicActions = piecePlacements[action].actList

            # debug piece placements                               
            #self.visualizePiecePositions(self.env,piecePlacements)
            
            # Display Loop
            reward = 0
            self.human_reward = 0
            for atomicAction in atomicActions:
                # make action
                observation, r, done, info = self.env.step(atomicAction)
                
                # accumulate reward
                reward += r
                
                # accumulate human feedback
                human_feedback = self.human_input.getAction()
                
                # check if escape key was pressed and teminate
                if human_feedback == -99: 
                    self.save_and_exit(training)
                
                if training:
                    self.human_reward += human_feedback
                
                if visualize:
                    # render to screen
                    self.env._render(tamer_feedback = human_feedback)
                
                    # pause a bit?
                    time.sleep(self.sleep_interval)
            
            if training:
                # apply backward update
                self.backward(self.prev_state_state_feats, self.prev_action, self.human_reward)

                # save previous action / states pair
                self.prev_action = copy.deepcopy(action)
                self.prev_state_state_feats = copy.deepcopy(state_state_feats)
            
            # save step infor after each step
            step_logs = {
                    'action': action,
                    'reward': reward,
                    }
            self.callbacks.on_step_end(episode_step, step_logs)
            episode_step += 1
            self.step+=1
            if self.step > max_episode_steps:
                done = True
        if not training:
            return episode_scores
    def forward(self, state_state_feats):
        
        # do a simple forward pass over each state-state diff
        pred_rewards = np.zeros((len(state_state_feats),))
        for idx, ss_feat in enumerate(state_state_feats):
            pred_rewards[idx] =self.model.predict_on_batch(np.reshape(ss_feat,(1,len(ss_feat))))
            
        # choose max
        action = np.random.choice(np.flatnonzero(pred_rewards == pred_rewards.max()))
        
        return action
    
    def backward(self, prev_state_state_feats, prev_action, human_reward):
        
        # do a gradient update step
        if human_reward != 0 and prev_state_state_feats is not None and prev_action is not None:
            x = np.reshape(prev_state_state_feats[prev_action],(1,len(prev_state_state_feats[prev_action]))).astype('float32')
            y = np.reshape(np.array(human_reward,dtype='float32'),(1))
            self.model.train_on_batch(x,y)
            self.nb_feedback_episode += 1
            self.nb_feedback_total += 1
            #print('model update')
    
    
class Deep_TAMERAgent(TetrisAgent):
    def __init__(self, model=None, AE = None, backprop_interval = 5, batch_size = 32, buffer_limit = 1000,
                 state_representation_method = 2, *args, **kwargs):
        super(Deep_TAMERAgent, self).__init__(*args, **kwargs)
        self.model = model
        self.AE = AE
        self.backprop_interval = backprop_interval
        self.batch_size = batch_size
        self.buffer_limit = buffer_limit
        self.state_representation_method = state_representation_method
        
        if self.AE == None:
            self.AE = SimpleAutoEncoder(type = 8)
            self.AE.load_weights('weights/autoencoder_8_Tetris-v0_weights.hf5')
        
        # experience replay buffer
        max_storage_limit = 500000
        self.feature_buffer = RingBuffer(max_storage_limit)
        self.mask_buffer = RingBuffer(max_storage_limit)
        self.target_buffer = RingBuffer(max_storage_limit)
        
    def compile(self, optimizer = None, learning_rate = 0.001, metrics=[]):
        # construct generic model if one is not specified
        if self.model == None:
            # Create the deep tamer model
            input_shape = (20,10,2)
            state_input = Input(shape = input_shape)
            x = Convolution2D(1,1,1, activation = 'elu')(state_input)
            x = Convolution2D(64,4,4, activation = 'elu',border_mode = 'same')(x)
            x = MaxPooling2D(pool_size=(2,2))(x)
            x = BatchNormalization()(x)
            x = Convolution2D(64,2,2, activation = 'elu',border_mode = 'same')(x)
            x = MaxPooling2D(pool_size=(2,2))(x)
            x = Flatten()(x)
            x = Dense(32, activation = 'elu')(x)
            x = BatchNormalization()(x)
            x = Dense(32, activation = 'elu')(x)
            pred_reward = Dense(1, init = 'zero')(x)
            self.model = Model(input = state_input, output = pred_reward)
            #self.model.summary()
        
        # compile the model
        if optimizer == None:
            self.model.compile(optimizer = Adam(lr=0.000005 / 47.0), loss='mse')
        else:
            self.model.compile(optimizer = optimizer, loss='mse')
        self.compiled = True
    
    def process_observation(self, observation):
        observation = observation.mean(2)
        processed_observation = observation.astype('float32') / 255.
        return processed_observation
    
    def fit(self, nb_steps=100000, callbacks = None, verbose = 3, visualize = True,
            training = True, model_file_name = 'Deep_Tetris-v0-weights.hf5'):
        if not training:
            # load model weights
            self.load_weights(model_file_name)
        else:
            if not self.compiled:
                raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        
        callbacks = [] if not callbacks else callbacks[:]
        self.callbacks = callbacks    
        self.callbacks = CallbackList(callbacks)
        if hasattr(self.callbacks, 'set_model'):
            self.callbacks.set_model(self)
        else:
            self.callbacks._set_model(self)
        self.callbacks._set_env(self.env)
        
        self.nb_steps = nb_steps
        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        done = True
        episode_scores = []
        while episode < self.nb_steps:
            
            if done:
                # save episode metrics after each episode (unless we are just starting)
                episode_reward = self.env.lines
                if not training:
                    episode_scores.append(episode_reward)
                if episode_step is not None:
                    episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            }
                    self.callbacks.on_episode_end(episode, episode_logs)
                print('Total Steps: {}, Episode Steps: {}, Total Episodes: {},  Lines Cleared: {}'.format(self.step,episode_step,episode,episode_reward))  
                episode_step = 0
                episode_reward = 0.
                episode +=1
                
                # Obtain the initial observation by resetting the environment.
                self.env.reset()
                if visualize:
                    self.env.render()
                
                self.prev_action = None
                self.prev_state_state_feats = None
            
            # for current piece, get all possible piece placements
            piecePlacements = self.getAllPiecePos(self.env)
            
            
            # extract state information  
            if self.state_representation_method == 1: # autoencoder state1 - state0 difference
                # for current state, extract features
                state0_feat = []
                state0_feat.append(self.process_observation(self.env.frame))
                state0_feat = np.array(state0_feat)
                state0_feat = state0_feat.reshape((state0_feat.shape+(1,)))
                state0_feat = self.AE.feature_extractor.predict_on_batch(state0_feat)
                
                # for each piece placement, extract features
                state1_feats = []
                for piecePlacement in piecePlacements:
                    self.setStateFromPos(piecePlacement)
                    self.tempState.draw()
                    state1_feats.append(self.process_observation(self.tempState.frame))
                state1_feats = np.array(state1_feats)
                state1_feats = state1_feats.reshape((state1_feats.shape + (1,)))
                state1_feats = self.AE.feature_extractor.predict_on_batch(state1_feats)
                
                # get state_state features
                state_state_feats = []
                for state1_feat in state1_feats:
                    state_state_feats.append(state1_feat - state0_feat)   
                    
            elif self.state_representation_method == 2:   # 20x10x2 raw begining and end state
                # for current state, extract features
                state0_feat = np.array(self.env.board)[0:20,:]
                
                # for each piece placement, extract features
                state1_feats = []
                for piecePlacement in piecePlacements:
                    state1_feats.append(np.array(piecePlacement.board)[0:20,:])
                    
                # get state_state features
                state_state_feats = []
                for state1_feat in state1_feats:
                    ss = np.concatenate((np.reshape(state0_feat,(state0_feat.shape + (1,))),
                                                             np.reshape(state1_feat,(state1_feat.shape + (1,)))),axis=2)
                    
                    state_state_feats.append(np.reshape(ss,((1,)+ss.shape)))   
                    
            else: # 1x200 raw end state
                # for current state, extract features
                state0_feat = np.reshape(np.array(self.env.board)[0:20,:],(1,200))
                
                # for each piece placement, extract features
                state1_feats = []
                for piecePlacement in piecePlacements:
                    state1_feats.append(np.reshape(np.array(piecePlacement.board)[0:20,:],(1,200)))
    
                # get state_state features
                state_state_feats = state1_feats
            
            # use tamer model to get best action
            action = self.forward(state_state_feats)
            
            # with final piece placement, step through all actions from start to finish
            # with some time delay, while also checking for human reward in that time
            # period to use as the label for the previous time block
            atomicActions = piecePlacements[action].actList

            # debug piece placements                               
            #self.visualizePiecePositions(self.env,piecePlacements)
            
            # Display Loop
            reward = 0
            self.human_reward = 0
            for atomicAction in atomicActions:
                # make action
                observation, r, done, info = self.env.step(atomicAction)
                
                # accumulate reward
                reward += r
                
                # accumulate human feedback
                human_feedback = self.human_input.getAction()
                
                # check if escape key was pressed and teminate
                if human_feedback == -99: 
                    self.save_and_exit(training)
                
                if training:
                    self.human_reward += human_feedback
                    
                if visualize:
                    # render to screen
                    self.env._render(tamer_feedback = human_feedback)
                
                    # pause a bit?
                    time.sleep(self.sleep_interval)
            
            if training:
                # apply backward update
                self.backward(self.prev_state_state_feats, self.prev_action, self.human_reward)
            
            # save step infor after each step
            step_logs = {
                    'action': action,
                    'reward': reward,
                    }
            self.callbacks.on_step_end(episode_step, step_logs)
            episode_step +=1
            self.step+=1
            
            # save previous action / states pair
            self.prev_action = copy.deepcopy(action)
            self.prev_state_state_feats = copy.deepcopy(state_state_feats)
        if not training:
            return episode_scores
        
    def forward(self, state_state_feats):
        
        # do a simple forward pass over each state-state diff
        pred_rewards = np.zeros((len(state_state_feats),))
        for idx, ss_feat in enumerate(state_state_feats):
            pred_rewards[idx] =self.model.predict_on_batch(ss_feat)
        #print(pred_rewards)
        # choose max
        action = np.random.choice(np.flatnonzero(pred_rewards == pred_rewards.max()))
        
        return action
    
    def backward(self, prev_state_state_feats, prev_action, human_reward):
        
        # do a gradient update step
        if human_reward != 0 and prev_state_state_feats is not None and prev_action is not None:
            self.nb_feedback_episode += 1
            self.nb_feedback_total += 1
            x = prev_state_state_feats[prev_action].astype('float32')
            y = np.reshape(np.array(human_reward,dtype='float32'),(1))
            self.model.train_on_batch(x,y)
            #print('model update')
            
            # save into buffer
            self.feature_buffer.append(prev_state_state_feats[prev_action])
            self.mask_buffer.append(prev_action)
            self.target_buffer.append(human_reward)
            
        if self.backprop_interval and self.step % self.backprop_interval == 0 and len(self.feature_buffer)> 2:
            # pull samples from the buffer
            r = xrange(0,min(self.buffer_limit-1,len(self.feature_buffer)-1))
            batch_idxs = random.sample(r,min(len(r),self.batch_size))
            
            features_batch = []
            targets = []
            masks = []
            for idx in batch_idxs:
                features_batch.append(self.feature_buffer[idx].astype('float32'))
                masks.append(self.mask_buffer[idx])
                targets.append(np.reshape(np.array(self.target_buffer[idx],dtype='float32'),(1)))
                
            features_batch = np.vstack(features_batch)
            targets = np.vstack(targets)
            masks = np.vstack(masks)
            
            self.model.train_on_batch(features_batch, targets)
            #print('replay update')
            
    def save_and_exit(self, training):
        if training:
            # save current model weights
            self.model.save_weights(self.save_file_name + '_weights.hf5')
        
            with open(self.save_file_name + '_feature_buffer.pkl','wb') as output:
                pickle.dump(self.feature_buffer, output, pickle.HIGHEST_PROTOCOL)        
            with open(self.save_file_name + '_target_buffer.pkl','wb') as output:
                pickle.dump(self.target_buffer, output, pickle.HIGHEST_PROTOCOL)            
            with open(self.save_file_name + '_mask_buffer.pkl','wb') as output:
                pickle.dump(self.mask_buffer, output, pickle.HIGHEST_PROTOCOL)        
            
        # exit
        exit(0)            
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)
        

class TAMER_Squared(Deep_TAMERAgent):
    def __init__(self, *args, **kwargs):
        super(TAMER_Squared, self).__init__(*args, **kwargs)
        
    def LHM_foward(self, state_state_feats):
        # do a simple forward pass over each state-state diff
        pred_rewards = np.zeros((len(state_state_feats),))
        for idx, ss_feat in enumerate(state_state_feats):
            pred_rewards[idx] = self.LHM.predict_on_batch(np.reshape(ss_feat,(1,len(ss_feat))))
        action = np.random.choice(np.flatnonzero(pred_rewards == pred_rewards.max()))
        return action
    
    def compile(self, optimizer = None, learning_rate = 0.001, LHM_weights = None):
        # construct generic model if one is not specified
        if self.model == None:
            
            # Create the deep tamer model
            input_shape = (20,10,2)
            state_input = Input(shape = input_shape)
            # compute before and (after-before) frames
            x0 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]))(state_input)
            x1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1] - x[:, :, :, 0]))(state_input)                     
            # concatenation
            x = Lambda(lambda x: K.concatenate([x[0], x[1]]))([x0, x1]) 
            # convolution
            c = Convolution2D(32, 4, 4, activation='relu', border_mode='same')(x)
            # flatten
            c = Flatten()(c) 
            # dense down to human reward 
            d = Dense(32, activation='relu')(c)
            d = Dense(32, activation='relu')(d)
            pred_reward = Dense(1)(d)
            self.model = Model(input = state_input, output = pred_reward)
            #self.model.summary()
        
        # create the pre-trained linear tamer model that will serve as the human
        # reward function
        input_shape = 46
        state_input = Input(shape = (input_shape,))
        pred_reward = Dense(1, init = 'zero')(state_input)
        self.LHM = Model(input = state_input, output = pred_reward)
        self.LHM.summary()
        self.LHM.compile(optimizer = SGD(lr=0.000005 / 47.0), loss='mse')
        if LHM_weights == None:
            self.LHM.load_weights('Vanilla_Tetris-v0-100-lines.hf5')
        else:
            self.LHM.load_weights(LHM_weights)
            
        # compile the model
        if optimizer == None:
            self.model.compile(optimizer = Adam(lr=0.001), loss='mse')
        else:
            self.model.compile(optimizer = optimizer, loss='mse')
        self.compiled = True

    def fit(self, nb_steps = 100000000, callbacks = None, verbose = 3, visualize = True):
        if not self.compiled:
            raise RuntimeError('Your tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        
        callbacks = [] if not callbacks else callbacks[:]
        self.callbacks = callbacks    
        self.callbacks = CallbackList(callbacks)
        if hasattr(self.callbacks, 'set_model'):
            self.callbacks.set_model(self)
        else:
            self.callbacks._set_model(self)
        self.callbacks._set_env(self.env)
        
        self.nb_steps = nb_steps
        episode = 0
        self.step = 0
        observation = None
        episode_reward = None
        episode_step = None
        done = True
        while episode < self.nb_steps:
            
            if done:
                # save episode metrics after each episode (unless we are just starting)
                episode_reward = self.env.lines
                if episode_step is not None:
                    episode_logs = {
                            'episode_reward': episode_reward,
                            'nb_episode_steps': episode_step,
                            'nb_steps': self.step,
                            }
                    self.callbacks.on_episode_end(episode, episode_logs)
                print('Total Steps: {}, Episode Steps: {}, Total Episodes: {},  Lines Cleared: {}'.format(self.step,episode_step,episode,episode_reward))  
                episode_step = 0
                episode_reward = 0.
                episode +=1
                
                # Obtain the initial observation by resetting the environment.
                self.env.reset()
                if visualize:
                    self.env.render()
            
            # for current piece, get all possible piece placements
            piecePlacements = self.getAllPiecePos(self.env)
            

            # Percieve state and determine action for deep tamer model            
            # for current state, extract features
            state0_feat = np.array(self.env.board)[0:20,:]
            # for each piece placement, extract features
            state1_feats = []
            for piecePlacement in piecePlacements:
                state1_feats.append(np.array(piecePlacement.board)[0:20,:])                
            # get state_state features
            state_state_feats = []
            for state1_feat in state1_feats:
                ss = np.concatenate((np.reshape(state0_feat,(state0_feat.shape + (1,))),
                                                         np.reshape(state1_feat,(state1_feat.shape + (1,)))),axis=2)
                state_state_feats.append(np.reshape(ss,((1,)+ss.shape)))   
                
            # use tamer model to get best action
            action = self.forward(state_state_feats)
            atomicActions = piecePlacements[action].actList
            
            # Percieve state and determine 'suggested' action from LHM
            s0_feat = self.env.getFeatures(self.env.board)
            s1_feats = []
            for piecePlacement in piecePlacements:
                s1_feats.append(self.env.getFeatures(piecePlacement.board))
                
            # get state_state features
            ss_feats = []
            for s1_feat in s1_feats:
                ss_feats.append(s1_feat - s0_feat)
                
            LHM_action = self.LHM_foward(ss_feats)
                
            # Get 'reward' from LHM
            if action == LHM_action:
                LHM_reward = 1
            else:
                LHM_reward = -1


            # Display Loop
            reward = 0
            for atomicAction in atomicActions:
                # make action
                observation, r, done, info = self.env.step(atomicAction)
                
                # accumulate reward
                reward += r
                
                # check if escape key was pressed and teminate
                key_pressed = self.human_input.getAction()
                if key_pressed == -99: 
                    self.save_and_exit(True)
                
                if key_pressed == 9:
                    visualize = not visualize
                # render to screen
                if visualize:
                    self.env._render(tamer_feedback = LHM_reward)
                    time.sleep(0.0)
            
            # apply backward update to deep tamer model
            self.backward(state_state_feats, action, LHM_reward)
            
            # save step infor after each step
            step_logs = {
                    'action': action,
                    'reward': reward,
                    }
            self.callbacks.on_step_end(episode_step, step_logs)
            episode_step +=1
            self.step+=1


if __name__ == '__main__':
    
    ## Debugging and practice script...
    
    # make tetris environment
    env = gym.make('Tetris-v0')
    
    # vanila tamer for tetris
    self = Super_Vanilla_TAMERAgent(model = None, env = env, sleep_interval = 0.0)
    self.compile()
    callbacks = [TensorBoard(log_dir='{}{}_{}_tensorboard_data'.format('TAMER_Experiments_aaai18/tamer_comparison/','vanilla_tamer','Tetris-v0'), agent = self)]               
    self.fit(nb_steps = 30, training = False, visualize = False, model_file_name = 'Vanilla_Tetris-v0-100-lines.hf5', callbacks = callbacks)
    
    # deep tamer for tetris
    self = Deep_TAMERAgent(model = None, env = env, sleep_interval = 0.0)
    self.compile()
    callbacks = [TensorBoard(log_dir='{}{}_{}_tensorboard_data'.format('TAMER_Experiments_aaai18/tamer_comparison/','deep_tamer','Tetris-v0'), agent = self)]               
    self.fit(nb_steps = 30, training = False, visualize = False, model_file_name = 'Tetris_weights/tamer2_Tetris-v0_weights.hf5', callbacks = callbacks)
        
    # tamer squared - tamer agent teaching another tamer agent
    self = TAMER_Squared(model = None, env = env, batch_size = 64, buffer_limit = 500000,
                         save_file_name = '{}{}_{}'.format('TAMER_Experiments_aaai18/tamer2/', 'tamer2', 'Tetris-v0'))
    self.compile()
    
    callbacks = [TensorBoard(log_dir='{}{}_{}_tensorboard_data'.format('TAMER_Experiments_aaai18/tamer2/','tamer2','Tetris-v0'), agent = self)]               
    self.fit(visualize = False, callbacks = callbacks)
    
    # load model weights
    self.model.save_weights('Vanilla_Tetris-v0-100-lines.hf5')
    
    

    
