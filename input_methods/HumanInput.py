#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:47:20 2017

@author: nicholas
"""

from input_methods import SimpleKeyboard
import threading
from configparser import SafeConfigParser


class HumanInput(object):
    def __init__(self,game, default_action = 0, input_method = 'keyboard', input_mode = 'TAMER'):
        
        self.inputMethod = input_method
        self.game = game
        self.action = 0
        self.default_action = default_action
        
        # input_mode: TAMER or playable
        self.input_mode = input_mode
        
        # create keyboard and start polling for keypresses
        self.keyboard = SimpleKeyboard.SimpleKeyboard()
        self.loadActionSpace_keyboard()
        
        
    def loadActionSpace_controller(self):
        # load in game specific control configuration
        cf_parser= SafeConfigParser()
        cf_parser.read('input_methods/ConfigHumanAgent.ini')
        game = self.game.split('-')[0]
        self.action_space=dict()
        self.action_space['lx_left'] = cf_parser.getint(game,'lx_left')
        self.action_space['lx_right'] = cf_parser.getint(game,'lx_right')
        self.action_space['ly_up'] = cf_parser.getint(game,'ly_up')
        self.action_space['ly_down'] = cf_parser.getint(game,'ly_down')
        self.action_space['rx_left'] = cf_parser.getint(game,'rx_left')
        self.action_space['rx_right'] = cf_parser.getint(game,'rx_right')
        self.action_space['ry_up'] = cf_parser.getint(game,'ry_up')
        self.action_space['ry_down'] = cf_parser.getint(game,'ry_down')
        self.action_space['hat0x_left'] = cf_parser.getint(game,'hat0x_left')
        self.action_space['hat0x_right'] = cf_parser.getint(game,'hat0x_right')
        self.action_space['hat0y_up'] = cf_parser.getint(game,'hat0y_up')
        self.action_space['hat0y_down'] = cf_parser.getint(game,'hat0y_down')
        self.action_space['select'] = cf_parser.getint(game,'select')
        self.action_space['start'] = cf_parser.getint(game,'start')        
        self.action_space['x'] = cf_parser.getint(game,'x')
        self.action_space['y'] = cf_parser.getint(game,'y')        
        self.action_space['b'] = cf_parser.getint(game,'b')
        self.action_space['a'] = cf_parser.getint(game,'a')
        self.action_space['mode'] = cf_parser.getint(game,'mode')
        self.action_space['tr'] = cf_parser.getint(game,'tr')
        self.action_space['tl'] = cf_parser.getint(game,'tl')
        self.action_space['rz'] = cf_parser.getint(game,'rz')
        self.action_space['lz'] = cf_parser.getint(game,'lz')
        self.action_space['thumbl'] = cf_parser.getint(game,'thumbl')
        self.action_space['thumbr'] = cf_parser.getint(game,'thumbr')
        
    def loadActionSpace_keyboard(self):
        # load in generic keyboard configuration for controlling games and 
        # providing tamer reward
        cf_parser = SafeConfigParser()
        cf_parser.read('input_methods/ConfigHumanAgent.ini')
        if self.input_mode == 'TAMER':
            game = 'Generic'
        else:
            game = self.game.split('-')[0]
        self.action_space=dict()
        self.action_space['1']=cf_parser.getint(game,'key_1')
        self.action_space['2']=cf_parser.getint(game,'key_2')
        self.action_space['3']=cf_parser.getint(game,'key_3')
        self.action_space['4']=cf_parser.getint(game,'key_4')
        self.action_space['5']=cf_parser.getint(game,'key_5')
        self.action_space['6']=cf_parser.getint(game,'key_6')
        self.action_space['7']=cf_parser.getint(game,'key_7')
        self.action_space['8']=cf_parser.getint(game,'key_8')
        self.action_space['9']=cf_parser.getint(game,'key_9')
        self.action_space['up']=cf_parser.getint(game,'key_up')  # default for positive tamer reward
        self.action_space['down']=cf_parser.getint(game,'key_down') # default for negative tamer reward
        self.action_space['left']=cf_parser.getint(game,'key_left')
        self.action_space['right']=cf_parser.getint(game,'key_right')
        self.action_space['esc']=cf_parser.getint(game,'key_esc')
        
        
    def getAction(self):
        self.action = self.default_action # default action
        
        # Joystick Control
        if(self.inputMethod=='joystick'):
            if(self.joyStick.axis_states['x']>=self.joyStick.stickTreshold): # right joystick
                self.action=self.action_space['lx_right'] 
            if(self.joyStick.axis_states['x']<=-self.joyStick.stickTreshold): # left joystick
                self.action=self.action_space['lx_left'] 
            if(self.joyStick.axis_states['y']<=-self.joyStick.stickTreshold): # up joystick
                self.action=self.action_space['ly_up']
            if(self.joyStick.axis_states['y']>=self.joyStick.stickTreshold): # down joystick
                self.action=self.action_space['ly_down']
            if(self.joyStick.axis_states['rx']>=self.joyStick.stickTreshold): 
                self.action=self.action_space['rx_right'] 
            if(self.joyStick.axis_states['rx']<=-self.joyStick.stickTreshold): 
                self.action=self.action_space['rx_left'] 
            if(self.joyStick.axis_states['ry']<=-self.joyStick.stickTreshold): 
                self.action=self.action_space['ry_up']
            if(self.joyStick.axis_states['ry']>=self.joyStick.stickTreshold): 
                self.action=self.action_space['ry_down']
            if(self.joyStick.axis_states['hat0x']>=self.joyStick.stickTreshold): 
                self.action=self.action_space['hat0x_right'] 
            if(self.joyStick.axis_states['hat0x']<=-self.joyStick.stickTreshold): 
                self.action=self.action_space['hat0x_left'] 
            if(self.joyStick.axis_states['hat0y']<=-self.joyStick.stickTreshold): 
                self.action=self.action_space['hat0y_up']
            if(self.joyStick.axis_states['hat0y']>=self.joyStick.stickTreshold): 
                self.action=self.action_space['hat0y_down']
            if(self.joyStick.axis_states['rz']>=self.joyStick.stickTreshold):
                self.action=self.action_space['rz']
            if(self.joyStick.axis_states['z']>=self.joyStick.stickTreshold):
                self.action=self.action_space['lz']
            # buttons
            if(self.joyStick.button_states['a']):
                self.action=self.action_space['a']
            if(self.joyStick.button_states['y']):
                self.action=self.action_space['y']
            if(self.joyStick.button_states['b']):
                self.action=self.action_space['b']
            if(self.joyStick.button_states['x']):
                self.action=self.action_space['x']
            if(self.joyStick.button_states['tr']):
                self.action=self.action_space['tr']
            if(self.joyStick.button_states['tl']):
                self.action=self.action_space['tl']
            if(self.joyStick.button_states['start']):
                self.action=self.action_space['start']
            if(self.joyStick.button_states['select']):
                self.action=self.action_space['select']
            if(self.joyStick.button_states['mode']):
                self.action=self.action_space['mode']
            if(self.joyStick.button_states['thumbl']):
                self.action=self.action_space['thumbl']
            if(self.joyStick.button_states['thumbr']):
                self.action=self.action_space['thumbr']
        
        # Keyboard Control
        if(self.inputMethod=='keyboard'):
            # standard action space
            if(self.keyboard.key_state['1']):
                self.action = self.action_space['1']
                self.keyboard.key_state['1'] = 0
            if(self.keyboard.key_state['2']):
                self.action = self.action_space['2']
                self.keyboard.key_state['2'] = 0                           
            if(self.keyboard.key_state['3']):
                self.action = self.action_space['3']
                self.keyboard.key_state['3'] = 0                          
            if(self.keyboard.key_state['4']):
                self.action = self.action_space['4']
                self.keyboard.key_state['4'] = 0                           
            if(self.keyboard.key_state['5']):
                self.action = self.action_space['5']
                self.keyboard.key_state['5'] = 0                           
            if(self.keyboard.key_state['6']):
                self.action = self.action_space['6']
                self.keyboard.key_state['6'] = 0                           
            if(self.keyboard.key_state['7']):
                self.action = self.action_space['7']
                self.keyboard.key_state['7'] = 0                           
            if(self.keyboard.key_state['8']):
                self.action = self.action_space['8']
                self.keyboard.key_state['8'] = 0                           
            if(self.keyboard.key_state['9']):
                self.action = self.action_space['9']
                self.keyboard.key_state['9'] = 0                           
            if(self.keyboard.key_state['up']):
                self.action = self.action_space['up']
                self.keyboard.key_state['up'] = 0                           
            if(self.keyboard.key_state['down']):
                self.action = self.action_space['down']
                self.keyboard.key_state['down'] = 0              
            if(self.keyboard.key_state['left']):
                self.action = self.action_space['left']
                self.keyboard.key_state['left'] = 0                           
            if(self.keyboard.key_state['right']):
                self.action = self.action_space['right']
                self.keyboard.key_state['right'] = 0                              
            if(self.keyboard.key_state['esc']):
                self.action = self.action_space['esc']
                self.keyboard.key_state['esc'] = 0
                                                           
        return self.action   