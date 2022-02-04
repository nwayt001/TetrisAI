#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:21:51 2017

@author: nicholas
"""
from pynput import keyboard
import threading

# Class for reading keyboard input
class SimpleKeyboard(object):
    """
    Simple Keyboard: reads what keys are recently pressed/released from on the keyboard
    """
    
    def __init__(self):
        self.debug_mode = False
        self.kb_thread = threading.Thread(target = self.pollKeyboard)
        self.kb_thread.start()
        
        self.key_state = {}
        self.key_state['1']=0
        self.key_state['2']=0
        self.key_state['3']=0
        self.key_state['4']=0
        self.key_state['5']=0
        self.key_state['6']=0              
        self.key_state['7']=0
        self.key_state['8']=0
        self.key_state['9']=0
        self.key_state['up']=0              
        self.key_state['down']=0     
        self.key_state['left']=0
        self.key_state['right']=0
        self.key_state['esc'] = 0   
                      
    def pollKeyboard(self):
        # Collect events until released
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join() 
    
    def on_press(self,key):
        if(self.debug_mode):
            print('{0} pressed'.format(key))
            
        # f12 terminates the listener and thread
        if key == keyboard.Key.f12:
            # stop listener
            return False
        
        # simple action space for controlling games manually
        if key == keyboard.KeyCode(char='1'):
            self.key_state['1'] = 1.0
        if key == keyboard.KeyCode(char='2'):
            self.key_state['2'] = 1.0
        if key == keyboard.KeyCode(char='3'):
            self.key_state['3'] = 1.0
        if key == keyboard.KeyCode(char='4'):
            self.key_state['4'] = 1.0
        if key == keyboard.KeyCode(char='5'):
            self.key_state['5'] = 1.0
        if key == keyboard.KeyCode(char='6'):
            self.key_state['6'] = 1.0
        if key == keyboard.KeyCode(char='7'):
            self.key_state['7'] = 1.0
        if key == keyboard.KeyCode(char='8'):
            self.key_state['8'] = 1.0
        if key == keyboard.KeyCode(char='9'):
            self.key_state['9'] = 1.0
        
        # Simple TAMER reward shaping (up-arrow means agent performed a good action and recieves a +1
        # down-arrow means agent performed a poor action and recieves a -1)
        if key == keyboard.Key.up:
            self.key_state['up'] = 1.0
        
        if key == keyboard.Key.down:
            self.key_state['down'] = 1.0
        
        if key == keyboard.Key.left:
            self.key_state['left'] = 1.0
        
        if key == keyboard.Key.right:
            self.key_state['right'] = 1.0
        
        
        if key == keyboard.Key.esc:
            self.key_state['esc'] = 1.0
            return False
    
    def on_release(self,key):
        if(self.debug_mode):
            print(' {0} released'.format(key))
            
        
        

if __name__ == '__main__':
    # Create a new keyboard reader
    kb = SimpleKeyboard()
    kb.debug_mode = True
    
