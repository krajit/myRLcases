import numpy as np
import matplotlib.pyplot as plt
import random
from IPython import display
from GridWorldEnv import GridWorldEnv
#from gridworldDynaTrainer import chooseAction

def chooseAction(Q,s,env,epsilon=0):
    if s not in Q: #seeing s for the first time
        Q[s] = {a:0. for a in range(env.action_space.n)}

    QsMax = max([Q[s][a] for a in range(env.action_space.n) ])
    aMax = []
    for a in range(env.action_space.n):
        if Q[s][a] >= QsMax - 1e-5:
            aMax.append(a)
    if np.random.rand() < epsilon:
        a = env.action_space.sample()
    else:
        a = np.random.choice(aMax)
    return a


import pickle
pickle_off = open("Qdict", 'rb')
Qdict = pickle.load(pickle_off)

Q = Qdict[8]

# test
maze = GridWorldEnv()
s = maze.reset()
s = tuple(s)
img = plt.imshow(maze.render()) 
plt.pause(0.1)
display.display(plt.gcf())
display.clear_output(wait=True)
done = False
while not done:
    a = chooseAction(Q,s,maze,epsilon=0.0)
    s, r, done, _ = maze.step(a)
    s = tuple(s)
    img.set_data(maze.render())
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.pause(0.1)