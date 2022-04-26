import argparse
import torch as th
import numpy as np
from tetris import TetrisEnv
from tetris_agent import Super_Vanilla_TAMERAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file',type=str,default='')
    parser.add_argument('--testing',type=bool,default=False)
    parser.add_argument('--practice',type=bool,default=False)
    parser.add_argument('--delay',type=float,default=0.0)
    args = parser.parse_args()      
    return args

if __name__ == '__main__':
    # get input args
    args = parse_args()


    # initialize tetris game gym environment
    env = TetrisEnv()
    
    # initialize tetris agent
    tetris_agent = Super_Vanilla_TAMERAgent(model = None, env = env, sleep_interval = 0.1)
    tetris_agent.compile()
    tetris_agent.fit()

    # save model
    tetris_agent.save_weights('trained_agent.hf5')