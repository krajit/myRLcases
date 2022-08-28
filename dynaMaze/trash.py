import gym
import numpy as np
from IPython import display
from matplotlib import pyplot as plt

def test_env(env: gym.Env) -> None:
    env.reset()
    done = False
    img = plt.imshow(env.render(mode='rgb_array')) 
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        img.set_data(env.render(mode='rgb_array')) 
        plt.axis('off')
        plt.pause(0.1)
        display.display(plt.gcf())
        display.clear_output(wait=True)


env = gym.make('CartPole-v1')
test_env(env)
env.close()