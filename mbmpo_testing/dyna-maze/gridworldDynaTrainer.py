import numpy as np
import matplotlib.pyplot as plt
import random
from IPython import display

epsilon = 0.1
alpha = 0.1
maxTol = 1e-5
gamma = 0.99

def chooseAction(Q,s,env,epsilon=0):
    if s not in Q: #seeing s for the first time
        Q[s] = {a:0. for a in range(env.action_space.n)}

    QsMax = max([Q[s][a] for a in range(env.action_space.n) ])
    aMax = []
    for a in range(env.action_space.n):
        if Q[s][a] >= QsMax - maxTol:
            aMax.append(a)
    if np.random.rand() < epsilon:
        a = env.action_space.sample()
    else:
        a = np.random.choice(aMax)
    return a

import pickle
pickling_on = open("Qdict","wb")


from GridWorldEnv import GridWorldEnv
stepsPerEpisode = {}
Qdict = {}
stepSizes = [0,1,8]
for stepSize in stepSizes:
    stepsPerEpisode[stepSize] = []
    env = GridWorldEnv()
    s = env.reset()
    Q = {}
    M = {} # model
    stepCounter = 0
    while len(stepsPerEpisode[stepSize]) < 100 : # outer loop    
        s = tuple(env.state)
        a = chooseAction(Q,s,env,epsilon=0.2)
        stepCounter += 1
        # take action
        sp, r, done, _ = env.step(a)
        sp = tuple(sp)
        if sp not in M:
            M[sp] = {}
        M[sp][a] = (r,sp) 
        # 
        aMax = chooseAction(Q,sp,env,epsilon=0.0)
        Q[s][a] += alpha*(r + gamma*Q[sp][aMax] - Q[s][a])
        for j in range(stepSize): #outerloop
            sj = random.choice([s for s in M])
            aj = random.choice([a for a in M[(sj)]])
            rj, sjp = M[sj][aj]
            maxQsjp = max([Q[sjp][a] for a in Q[sjp]])
            Q[sj][aj] += alpha*(rj + gamma*maxQsjp - Q[sj][aj]) 

        if done:
            stepsPerEpisode[stepSize].append(stepCounter)
            stepCounter = 0
            env.reset()
    Qdict[stepSize] = Q


pickle.dump(Qdict, pickling_on)
pickling_on.close()

for stepSize in stepSizes:
    plt.plot(stepsPerEpisode[stepSize])
plt.legend([str(s) for s in stepSizes])
plt.show()



