import gym
import numpy as np
import random
import matplotlib.pylab as plt
import copy
import time

from typing import List, Optional

def newGenerate_random_map(size: int = 8, p: float = 0.8) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    # DFS to check that it's a valid path.
    def is_valid(res,sx,sy):
        frontier, discovered = [], set()
        frontier.append((sx, sy))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        sx = np.random.randint(size); sy = np.random.randint(size)
        res[sx][sy] = "S"
        gx = np.random.randint(size); gy = np.random.randint(size)
        while res[gx][gy] == "S": # we don't want to overwrite the S
            gx = np.random.randint(size); gy = np.random.randint(size)
        res[gx][gy] = "G"   
        valid = is_valid(res,sx,sy)
    return ["".join(x) for x in res]