'''
Random Agets to explore the environment? Create Value table???
'''

import numpy as np
import random

from game import *


class Greedy_Agent():
    def __init__(self, starting_pos:tuple[int,int], n:int=9, init_value_table:np.ndarray=None, decay:float=.01, steps:int=0):
        self.value_table = init_value_table if init_value_table is not None else np.array([[0 for _ in range(n)] for _ in range(n)]) 
        self.count_table = np.array([[0 for _ in range(n)] for _ in range(n)])
        self.pos = starting_pos
        self.prev_pos = []

        self.decay = decay
        self.steps = steps

    def reward_step(self):
        for p in self.prev_pos: 
            x,y = p
            self.value_table[x-1,y-1] += 1 #Serves as both a count and a value (), env has boarder

    def action(self, board_state: np.ndarray) -> Move:
        #, pos: tuple[int, int]) -> Move:

        # Inc
        self.prev_pos.append(self.pos) 
        self.reward_step()
        self.count_table[self.pos[0]-1, self.pos[1]-1] += 1

        x, y = self.pos

        valid_moves = []

        if board_state[x+1][y] == 0:
            valid_moves.append((Move.RIGHT, self.value_table[x][y-1]))
        if board_state[x-1][y] == 0:
            valid_moves.append((Move.LEFT, self.value_table[x-2][y-1])) 
        if board_state[x][y+1] == 0:
            valid_moves.append((Move.UP, self.value_table[x-1][y]))
        if board_state[x][y-1] == 0:
            valid_moves.append((Move.DOWN, self.value_table[x-1][y-2])) 

        eps = .01 * (1. - .01) * np.exp(-1. * self.decay * self.steps)
        if len(valid_moves) > 0:
            if random.random() < eps:
                rm = random.choice(valid_moves)[0]
            else:
                rm = valid_moves[-1][0]
        else:
            rm = random.choice(list(Move))
        print(rm) 
        return rm


def play_game(n:int, iv1, iv2, steps) -> np.ndarray:
    
    env = SnakeGame(n)
    ra1 = Greedy_Agent(env.pos1, n, iv1, steps=steps)
    ra2 = Greedy_Agent(env.pos2, n, iv2, steps=steps)

    while True:
        # Player 1
        move1 = ra1.action(env.state)

        # Player 2
        move2 = ra2.action(env.state)

        # Update the environment
        s, r, done = env.step(move1, move2)

        # Update the agents
        ra1.pos = s[0]
        ra2.pos = s[1]

        if done:
            print(f'Got reward: {r}')
            steps += r
            break

    return ra1, ra2


if __name__ == '__main__':
    '''
    Just going to create a value table for the environment
    Then I create an agent that moves in the optimal direction
    '''

    # Make the environment 
    grid_size = 9 
    a1s = []
    a2s = []

    v1 = np.array([[0 for _ in range(grid_size)] for _ in range(grid_size)])
    v2 = np.array([[0 for _ in range(grid_size)] for _ in range(grid_size)])
    steps = 0

    for _ in range(int(1e1)):
        a1, a2 = play_game(grid_size, v1, v2, steps)
        a1s.append(a1)
        a2s.append(a2)

        v1 = np.divide((v1 + a1.value_table + 1e-6), (a1.count_table + np.ones_like(v1) + 1e-6))
        v2 = np.divide((v2 + a2.value_table + 1e-6), (a2.count_table + np.ones_like(v1) + 1e-6))

    # Make value matrix for each state (average over all games played), will choose state with the most value
    a1_avg_value = np.divide((np.stack([a.value_table for a in a1s]) + 1e-6), (np.stack([a.count_table for a in a1s]) + 1e-6)).mean(axis=0)
    a2_avg_value = np.divide((np.stack([a.value_table for a in a2s]) + 1e-6), (np.stack([a.count_table for a in a2s]) + 1e-6)).mean(axis=0)

    #np.save('a1_avg_value.npy', a1_avg_value)
    #np.save('a2_avg_value.npy', a2_avg_value)