#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:00:53 2017

Tetris implementation for the gym environment

@author: nicholas waytowich
"""

from random import randrange as rand
import pygame
import sys
import gym
import numpy as np
from gym import spaces
import copy
from PIL import Image as ImageUtil #*nrw-edit*


class TetrisEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, state_mode='pixels'):
        # game configuration
        self.paused = False
        self.gameover = False
        self.cell_size = 22
        self.border_size = 19
        self.cols = 10
        self.rows = 20
        self.maxfps = 30
        self.viewer = None
        self.state_mode = state_mode
        self.height_multiplier = 2
        self.width_multiplier = 2
        self._seed = 1337

        self.colors = [
            (0,   0,   0),
            (255, 85,  85),
            (100, 200, 115),
            (120, 108, 245),
            (255, 140, 50),
            (50,  120, 52),
            (146, 202, 73),
            (150, 161, 218),
            (35,  35,  35)  # Helper color for background grid
        ]

        # define shapes of the different tetris blocs
        self.tetris_shapes = [
            [[1, 1, 1],
             [0, 1, 0]],

            [[0, 2, 2],
             [2, 2, 0]],

            [[3, 3, 0],
             [0, 3, 3]],

            [[4, 0, 0],
             [4, 4, 4]],

            [[0, 0, 5],
             [5, 5, 5]],

            [[6, 6, 6, 6]],

            [[7, 7],
             [7, 7]]
        ]
        # action space
        self.perform_action = {
            0: lambda: self.move(-1),
            1: lambda: self.move(+1),
            2:	self.rotate_stone_cw,
            3:	self.rotate_stone_ccw,
            4:	self.no_op,
            5:	self.insta_drop}
        self.action_space = spaces.Discrete(len(self.perform_action)-1)

        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.width = self.cell_size*(self.cols+6)
        self.height = self.cell_size*self.rows
        self.rlim = self.cell_size*self.cols
        self.bground_grid = [[8 if x % 2 == y % 2 else 8 for x in range(
            self.cols)] for y in range(self.rows)]
        self.default_font = pygame.font.Font(
            pygame.font.get_default_font(), 12)
        self.screen = pygame.Surface((self.width, self.height))
        #self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # Block mouse events
        self.nextStoneId = rand(len(self.tetris_shapes))
        self.next_stone = self.tetris_shapes[self.nextStoneId]

        # Game State
        self.board = []
        self.stone = []
        self.currentStoneId = []
        self.currentRotation = []
        self.stone_x = 0
        self.stone_y = 0
        self.blockMobile = True

        # initialize new game
        self.init_game()
        self.obs = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        self.frame = np.zeros(
            (self.cell_size*self.cols, self.height, 3), dtype=np.uint8)
        pygame.pixelcopy.surface_to_array(self.obs, self.screen)

        # Hand-crafted Tetris Features
        self.NUM_FEATS = 46
        self.COL_HT_START_I = 0
        self.MAX_COL_HT_I = 10
        self.COL_DIFF_START_I = 11
        self.NUM_HOLES_I = 20
        self.MAX_WELL_I = 21
        self.SUM_WELL_I = 22
        self.SQUARED_FEATS_START_I = 23
        self.SCALE_ALL_SQUARED_FEATS = False
        self.HT_SQ_SCALE = 100.0

    def rotate_clockwise(self, shape):
        return [[shape[y][x]
                 for y in range(len(shape))]
                for x in range(len(shape[0]) - 1, -1, -1)]

    def rotate_counterclockwise(self, shape):
        # this is a hack to do a CCW rotation. it's probably not efficient...
        for i in range(3):
            shape = [[shape[y][x]
                      for y in range(len(shape))]
                     for x in range(len(shape[0]) - 1, -1, -1)]
        return shape

    def check_collision(self, board, shape, offset):
        off_x, off_y = offset
        for cy, row in enumerate(shape):
            for cx, cell in enumerate(row):
                try:
                    if cell and board[cy + off_y][cx + off_x]:
                        return True
                except IndexError:
                    return True
        return False

    def remove_row(self, board, row):
        del board[row]
        return [[0 for i in range(self.cols)]] + board

    def join_matrices(self, mat1, mat2, mat2_off):
        off_x, off_y = mat2_off
        for cy, row in enumerate(mat2):
            for cx, val in enumerate(row):
                mat1[cy + off_y-1][cx+off_x] += val
        return mat1

    def new_board(self):
        board = [[0 for x in range(self.cols)]
                 for y in range(self.rows)]
        board += [[1 for x in range(self.cols)]]
        return board

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.currentStoneId = self.nextStoneId
        self.currentRotation = 0
        self.nextStoneId = rand(len(self.tetris_shapes))
        self.next_stone = self.tetris_shapes[self.nextStoneId]
        self.stone_x = int(self.cols / 2 - len(self.stone[0])/2)
        self.stone_y = 0
        self.blockMobile = True
        if self.check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True
            self.blockMobile = False

    def init_game(self):
        self.board = self.new_board()
        self.new_stone()
        self.level = 1
        self.score = 0
        self.prev_score = 0
        self.lines = 0

        pygame.time.set_timer(pygame.USEREVENT+1, 1000)

    def disp_msg(self, msg, topleft):
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                            (255, 255, 255),
                            (0, 0, 0)),
                (x, y))
            y += 14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False,
                                                 (255, 255, 255), (0, 0, 0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
                self.width // 2-msgim_center_x,
                self.height // 2-msgim_center_y+i*22))

    def draw_matrixOrig(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        self.colors[val],
                        pygame.Rect(
                            (off_x+x) *
                            self.cell_size,
                            (off_y+y) *
                            self.cell_size,
                            self.cell_size,
                            self.cell_size), 0)

    def draw_matrix(self, matrix, offset):
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    # draw border
                    pygame.draw.rect(
                        self.screen,
                        self.colors[val],
                        pygame.Rect(
                            (off_x+x) *
                            self.cell_size,
                            (off_y+y) *
                            self.cell_size,
                            self.cell_size,
                            self.cell_size), 0)

    def add_cl_lines(self, n):
        linescores = [0, 40, 100, 300, 1200]
        self.lines += n
        self.score += linescores[n] * self.level
        if self.lines >= self.level*6:
            self.level += 1
            newdelay = 1000-50*(self.level-1)
            newdelay = 100 if newdelay < 100 else newdelay
            pygame.time.set_timer(pygame.USEREVENT+1, newdelay)

    def move(self, delta_x):
        outofbounds = False
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
                outofbounds = True
            if new_x > self.cols - len(self.stone[0]):
                new_x = self.cols - len(self.stone[0])
                outofbounds = True
            if not self.check_collision(self.board,
                                        self.stone,
                                        (new_x, self.stone_y)):
                self.stone_x = new_x
                collision = False
            else:
                collision = True

        return outofbounds, collision

    def quit(self):
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self, manual):
        if not self.gameover and not self.paused:
            self.score += 1 if manual else 0
            self.stone_y += 1
            if self.check_collision(self.board,
                                    self.stone,
                                    (self.stone_x, self.stone_y)):
                self.board = self.join_matrices(
                    self.board,
                    self.stone,
                    (self.stone_x, self.stone_y))
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = self.remove_row(
                                self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.add_cl_lines(cleared_rows)
                return True
        return False

    def no_op(self):
        pass

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while(not self.drop(True)):
                pass

    def rotate_stone_cw(self):
        if not self.gameover and not self.paused:
            new_stone = self.rotate_clockwise(self.stone)
            if not self.check_collision(self.board,
                                        new_stone,
                                        (self.stone_x, self.stone_y)):
                self.currentRotation = (self.currentRotation + 1) % 4
                self.stone = new_stone
                return True
            else:
                return False

    def rotate_stone_ccw(self):
        if not self.gameover and not self.paused:
            new_stone = self.rotate_counterclockwise(self.stone)
            if not self.check_collision(self.board,
                                        new_stone,
                                        (self.stone_x, self.stone_y)):
                self.currentRotation = self.currentRotation - 1
                if self.currentRotation < 0:
                    self.currentRotation = 3
                self.stone = new_stone
                return True
            else:
                return False

    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    def reset(self, episode = 1):
        self.init_game()
        self.level = episode
        self.gameover = False
        self.draw()

        if self.state_mode == 'pixels':
            return self.frame
        elif self.state_mode == 'observation':
            return self.getObservation()

    def getObservation(self):
        observation = np.zeros((207,))
        observation[0:200] = np.reshape(
            np.array(self.board)[0:20, :], (1, 200))
        observation[200] = 0 if self.check_collision(
            self.board, self.stone, (self.stone_x, self.stone_y)) else 1
        observation[201] = self.currentStoneId
        observation[202] = self.currentRotation
        observation[203] = self.stone_x
        observation[204] = self.stone_y
        observation[205] = self.cols
        observation[206] = self.rows

        return observation

    def getFeatures(self, board):
        featsArray = np.zeros(46,) - 1
        featsArray[self.NUM_HOLES_I] = 0
        featsArray[self.SUM_WELL_I] = 0
        board = np.array(board)
        # get column heights/ max column height and number of holes features
        for row in range(self.rows+1):
            for col in range(self.cols):
                if(board[row, col] > 0):  # filled cell
                    if(featsArray[self.COL_HT_START_I + col] == -1):
                        featsArray[self.COL_HT_START_I + col] = row
                    if(featsArray[self.MAX_COL_HT_I] == -1):
                        featsArray[self.MAX_COL_HT_I] = row
                else:  # empty cell
                    if (featsArray[self.COL_HT_START_I + col] != -1):
                        featsArray[self.NUM_HOLES_I] += 1
        if(featsArray[self.MAX_COL_HT_I] == -1):
            featsArray[self.MAX_COL_HT_I] = self.rows

        # get column difference features
        for col in range(self.cols - 1):
            featsArray[self.COL_DIFF_START_I + col] = abs(
                featsArray[self.COL_HT_START_I + col] - featsArray[self.COL_HT_START_I + col + 1])

        # get well depth features
        for col in range(self.cols):
            wellDepth = self.getWellDepth(col, board)
            featsArray[self.SUM_WELL_I] += wellDepth
            if wellDepth > featsArray[self.MAX_WELL_I]:
                featsArray[self.MAX_WELL_I] = wellDepth

        # get squared features and scale them so they're not too big
        for i in range(self.SQUARED_FEATS_START_I):
            featsArray[self.SQUARED_FEATS_START_I +
                       i] = np.square(featsArray[i])
            if(i <= self.MAX_COL_HT_I or self.SCALE_ALL_SQUARED_FEATS):
                featsArray[self.SQUARED_FEATS_START_I + i] /= self.HT_SQ_SCALE

        return featsArray

    def getWellDepth(self, col, board):
        depth = 0
        for row in range(self.rows):
            if(board[row, col] > 0):  # encountered a filled space, stop counting
                return depth
            else:
                if depth > 0:  # if well-depth count has begun, dont require left or right side to be filled
                    depth += 1
                # check if both the cell to the left if full and if the cell to the right is full
                elif (col == 0 or board[row, col-1] > 0) and (col == self.cols-1 or board[row, col+1] > 0):
                    depth += 1
        return depth

    # draw current board and piece position in buffer

    def draw(self):
        self.screen.fill((0, 0, 0))
        if self.gameover:
            self.center_msg("""Game Over!\nYour score: %d 
                Press space to continue""" % self.score)
        else:
            if self.paused:
                self.center_msg("Paused")
            else:
                pygame.draw.line(self.screen,
                                 (255, 255, 255),
                                 (self.rlim+1, 0),
                                 (self.rlim+1, self.height-1))
                self.disp_msg("Next:", (
                    self.rlim+self.cell_size,
                    2))
                self.disp_msg("Score: %d\n\nLevel: %d \nLines: %d" % (self.score, self.level, self.lines),
                              (self.rlim+self.cell_size, self.cell_size*5))
                self.draw_matrix(self.bground_grid, (0, 0))
                self.draw_matrix(self.board, (0, 0))
                self.draw_matrix(self.stone,
                                 (self.stone_x, self.stone_y))
                self.draw_matrix(self.next_stone,
                                 (self.cols+1, 2))
                # copy contents from pygame surface to observation buffer
                pygame.pixelcopy.surface_to_array(self.obs, self.screen)
                self.frame = self.obs[:self.cell_size*self.cols, :, :]

    # used to take a 'simulated' action for tree-searching
    def take_action(self, action):
        legalMove = False

        # perform action
        if action == 0:  # move left
            legalMove = self.move(-1)

        elif action == 1:  # move right
            legalMove = self.move(1)

        elif action == 2:  # move clockwise
            legalMove = self.rotate_stone_cw()

        elif action == 3:  # move counterclockwise
            legalMove = self.rotate_stone_ccw()

        # else no-op action, do nothing
        elif action == 4:
            legalMove = True

        # now check if resulting position is legal, if so keep it, otherwise don't change anything
        return legalMove

    # update and draw 'simulated' action with a drop of one position
    def update_and_draw(self):
        board = self.board
        self.stone_y += 1
        if self.check_collision(self.board,
                                self.stone,
                                (self.stone_x, self.stone_y)):
            self.blockMobile = False
            board = self.join_matrices(
                self.board,
                self.stone,
                (self.stone_x, self.stone_y))
            self.stone_y -= 1

        self.draw()

        return board

    # update 'simulated' action with a drop of one position
    def update(self):
        board = copy.deepcopy(self.board)
        self.stone_y += 1
        if self.check_collision(self.board,
                                self.stone,
                                (self.stone_x, self.stone_y)):
            self.blockMobile = False
            m1 = copy.deepcopy(self.board)
            m2 = copy.deepcopy(self.stone)
            m2_off = copy.deepcopy((self.stone_x, self.stone_y))
            board = self.join_matrices(m1, m2, m2_off)
            self.stone_y -= 1

        return board

    # take step in real environment
    def step(self, action):
        # perform action
        self.perform_action[action]()
        self.drop(False)

        # update and draw to screen
        self.draw()

        # update reward and score
        self.reward = self.score - self.prev_score
        self.prev_score = self.score

        # return observation, reward, done, info
        if self.state_mode == 'pixels':
            return self.frame, self.reward, self.gameover, {}
        elif self.state_mode == 'observation':
            return self.getObservation(), self.reward, self.gameover, {}

    # render to game screen
    def render(self, mode='human', tamer_feedback=None, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = np.transpose(self.obs, axes=(1, 0, 2))

        if tamer_feedback == 1:
            img[1:self.cell_size*2, 1:self.cell_size*2, :] = 0
            img[1:self.cell_size*2, 1:self.cell_size*2, 1] = 255
        elif tamer_feedback == -1:
            img[1:self.cell_size*2, 1:self.cell_size*2, :] = 0
            img[1:self.cell_size*2, 1:self.cell_size*2, 0] = 255
        elif tamer_feedback == 0:
            img[1:self.cell_size*2, 1:self.cell_size*2, :] = 0

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(
                    maxwidth=1000)

            # scale image to make it larger
            im = ImageUtil.fromarray(img, 'RGB')
            im = im.resize((img.shape[1]*self.width_multiplier, img.shape[0]*self.height_multiplier))
            img = np.array(im)

            # show image
            self.viewer.imshow(img)

        elif mode == 'compact_state':
            return [self.currentStoneId, self.stone_x, self.stone_y, self.score,
                    self.gameover, self.cols, self.rows, self.board]


'''
# Example usage
game = Tetris()
es = game.step(1)
game.render()

# action space
# action space
        self.action_space = {
			0:  self.no_op,
			1:	lambda:self.move(-1),
			2:	lambda:self.move(+1),
			3:	lambda:self.drop(True),
			4:	self.rotate_stone,
			5:	self.insta_drop}

'''
