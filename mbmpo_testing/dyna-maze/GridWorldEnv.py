import numpy as np
import matplotlib.pyplot as plt
import pygame
from IPython import display

ROWS = 6
COLS = 9
S = np.array([0,0])
G = np.array([0,8])

#BLOCKS = [(1, 2), (2, 2), (3, 2), (0, 7), (2, 7), (4, 5)]
BLOCKS = np.array([[1, 2], [2, 2], [3, 2], [0, 7],[1,7], [2, 7],[3,7], [4, 5]])
ACTIONS = ["left", "up", "right", "down"]

import gym
from gym import spaces
import numpy as np
from typing import Optional

class GridWorldEnv(gym.Env):
    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.start = S
        self.goal = G
        self.blocks = BLOCKS
        self.state = S
        self.end = False
        #init maze
        self.observation_space = spaces.MultiDiscrete([ROWS,COLS])

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),  # down
            1: np.array([0, 1]),  # right
            2: np.array([-1, 0]), # up
            3: np.array([0, -1]), # left
        }


    def step(self, action):
        reward = 0
        done = False        
        (r,c) = self.state + self._action_to_direction[action]
        if (r >= 0 and r <= self.rows-1) and (c >= 0 and c <= self.cols-1):
            if [r,c] not in self.blocks.tolist():
                self.state = np.array([r,c])
        
        if (self.state == self.goal).all():
            done = True
            reward = 1

        return self.state, reward, done, {}

    def reset(self):
        self.state = S
        self.end = False
        return self.state


    def giveReward(self):
        if (self.state == self.goal).all():
            print("goal reached")
            self.end = True
            return 1
        else:
            return 0

    def render(self):
        sqSize = 20
        canvasRowSize = sqSize*self.rows+1
        canvasColSize = sqSize*self.cols+1
        canvas = pygame.Surface((canvasRowSize,canvasColSize))
        canvas.fill((255,255,255))
        for i in range(self.cols+2):
            pygame.draw.line(canvas, 0, (0, sqSize*i), (canvasRowSize, sqSize*i), width=1)
        for i in range(self.rows+2):
            pygame.draw.line(canvas, 0, (sqSize*i,0), (sqSize*i,canvasColSize), width=1)

        r,c = self.state
        blueColor = (0,0,255)
        pygame.draw.circle(canvas, blueColor, (r*sqSize+sqSize/2,c*sqSize+sqSize/2) , sqSize/3)

        r,c = self.goal
        greenColor = (0,255,0)
        pygame.draw.circle(canvas, greenColor, (r*sqSize+sqSize/2,c*sqSize+sqSize/2) , sqSize/3)

        blackColor = (0,0,0)
        for (r,c) in self.blocks:
            pygame.draw.rect(canvas, blackColor, pygame.Rect(sqSize*r+1, sqSize*c+1, sqSize-1, sqSize-1))

        plArray = np.array(pygame.surfarray.pixels3d(canvas))
        return plArray




if __name__ == "__main__":
    maze = GridWorldEnv()
    maze.reset()
    img = plt.imshow(maze.render()) 
    plt.pause(0.1)
    display.display(plt.gcf())
    display.clear_output(wait=True)
    done = False
    while not done:
        a = maze.action_space.sample()
        s, r, done, _ = maze.step(a)
        img.set_data(maze.render())
        display.display(plt.gcf())
        display.clear_output(wait=True)
        plt.pause(0.1)
