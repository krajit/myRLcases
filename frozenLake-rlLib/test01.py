import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer

import pygame
import numpy as np

import torch
from IPython.display import clear_output
from IPython import display
import random
import matplotlib.pylab as plt
import copy
import time

from typing import List, Optional
from gym.envs.toy_text.frozen_lake import generate_random_map

from newGenerate_random_map import newGenerate_random_map
ray.init(local_mode=True, num_gpus=0)

class WrappedFrozenLake(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.desc = self.env.desc
        self.size = len(self.desc)
        self.old = np.where((env.desc==b'S').reshape(self.size**2,))[0]
        self.goal = np.where((env.desc==b'G').reshape(self.size**2,))[0]
        self.observation_space = gym.spaces.Box(-0.1,1.1,shape=(3,self.size,self.size,),dtype="float32")
        self.action_space = self.env.action_space
        self.state_ = None
        
    def oneHot(self,s):
        x = np.zeros(self.size*self.size)
        x[s] = 1
        state_ = np.array([ x.reshape(self.size,self.size),
                         np.array(self.env.desc == b"F").astype("float32"),
                         np.array(self.env.desc == b"G").astype("float32")
                          ])
        return state_.reshape(3,self.size,self.size,) + (0.1*np.random.rand(3,self.size,self.size,)-0.05)

    def reset(self):
        # return self.oneHot(1)
        self.s = self.old
        self.state_ = self.oneHot(self.env.reset())
        
class WrappedFrozenLake(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.desc = self.env.desc
        self.size = len(self.desc)
        self.old = np.where((env.desc==b'S').reshape(self.size**2,))[0]
        self.goal = np.where((env.desc==b'G').reshape(self.size**2,))[0]
        self.observation_space = gym.spaces.Box(-0.1,1.1,shape=(3,self.size,self.size,),dtype="float32")
        self.action_space = self.env.action_space
        self.state_ = None
        
    def oneHot(self,s):
        x = np.zeros(self.size*self.size)
        x[s] = 1
        state_ = np.array([ x.reshape(self.size,self.size),
                         np.array(self.env.desc == b"F").astype("float32"),
                         np.array(self.env.desc == b"G").astype("float32")
                          ])
        return state_.reshape(3,self.size,self.size,) + (0.1*np.random.rand(3,self.size,self.size,)-0.05)

    def reset(self):
        # return self.oneHot(1)
        self.s = self.old
        self.state_ = self.oneHot(self.env.reset())
        return self.state_
    
    def step(self, action):
        obs, r, done, info = s
        return self.state_
    
    def step(self, action):
        obs, r, done, info = self.env.step(action)
        
        if obs == self.goal:
            done = True
            
        if done:
            reward = 2 if r > 0 else -1
        elif obs == self.old:
            reward = -1
        else:
            reward = 0
        self.old = obs
        self.state_ = self.oneHot(obs)
        return self.state_, reward, done, info
    
    def render(self, mode="rgb_array"):
        sqSize = 20
        canvasSize = sqSize*self.size+1
        canvas = pygame.Surface((canvasSize, canvasSize))
        canvas.fill((255, 255, 255))
        for i in range(self.size+2):
            pygame.draw.line(canvas, 0, (0, sqSize*i), (canvasSize, sqSize*i), width=1)
            pygame.draw.line(canvas, 0, ( sqSize*i,0), (sqSize*i,canvasSize, ), width=1)
        for i in range(self.size):
            for j in range(self.size):
                if (self.env.desc==b"H")[i,j]: # if there is a hole at ij
                    blackColor = (0,0,0)
                    pygame.draw.rect(canvas, blackColor, pygame.Rect(sqSize*(i)+1, sqSize*(j)+1, sqSize-1, sqSize-1))
                if (self.env.desc==b"G")[i,j]: # if there is a gole at ij
                    greenColor = (0,255,0)
                    pygame.draw.rect(canvas, greenColor, pygame.Rect(sqSize*(i), sqSize*(j), sqSize, sqSize))
                if self.state_[0,i,j] > 0.5:
                    blueColor = (0,0,255)
                    pygame.draw.circle(canvas, blueColor, (sqSize*i+sqSize/2,sqSize*j+sqSize/2) , sqSize/3)
        
        plArray = np.array(pygame.surfarray.pixels3d(canvas))
        plt.imshow(plArray)        
        plt.axis("off")

from ray.tune.registry import register_env

def env_creator(env_config): 
    size = env_config['size']
    numHoles = env_config['numHoles']
    p = 1-numHoles/(size**2)
    desc = newGenerate_random_map(size=size, p=p)
    return WrappedFrozenLake(gym.make('FrozenLake-v1', desc = desc, is_slippery=False))  # return an env instance

register_env("myenv", env_creator)

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()

class MyTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.device = torch.device("cpu")#"cuda"
        #                            if torch.cuda.is_available() else "cpu")

        self.mainLayer = nn.Sequential(
                nn.Conv2d(3,12,kernel_size=3,stride= 1,padding=1),
                nn.ReLU(),
                nn.Conv2d(12,24,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(24,36,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),            
                nn.Flatten(start_dim=-3,end_dim=-1),
                nn.Linear(1296,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
        ).to(self.device)
        
        # Action logits output.
        self.layer_out = nn.Linear(256, num_outputs).to(self.device)

        # "Value"-branch (single node output).
        # Used by several RLlib algorithms (e.g. PPO) to calculate an observation's value.
        self.value_branch = nn.Linear(256, 1).to(self.device)
        self.cur_value = None

    def forward(self, input_dict, state, seq_lens):
        """Custom-define your forard pass logic here."""
        # Pass inputs through our 2 layers.
        layer_1_out = self.mainLayer(input_dict["obs"])
        logits = self.layer_out(layer_1_out)

        # Calculate the "value" of the observation and store it for
        # when `value_function` is called.
        self.cur_value = self.value_branch(layer_1_out).squeeze(-1)

        return logits, state

    def value_function(self):
        """Implement the value branch forward pass logic here:
        
        We will just return the already calculated `self.cur_value`.
        """
        assert self.cur_value is not None, "Must call `forward()` first!"
        return self.cur_value


config = {
        "framework": "torch",
        "env":"myenv",  
        "env_config":{'size':6, 'numHoles': 10},
        "num_workers": 1,
          "model": {
             "custom_model": MyTorchModel,  # for torch users: "custom_model": MyTorchModel
             "custom_model_config": {},
          },
        'num_envs_per_worker': 200,
        "create_env_on_driver": True,
}

# # Create our RLlib Trainer.
trainer = PPOTrainer(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(10):
   analysis =  trainer.train()
   print(analysis['sampler_results']['episode_reward_mean'])
