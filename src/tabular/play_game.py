'''
Random Agets to explore the environment? Create Value table???
'''

import numpy as np
import random

from game import *

class Random_Agent():
    def __init__(self, value_matrix:np.ndarray, starting_pos:tuple[int,int], n:int=9):
        self.pos = starting_pos
        self.value_matrix = value_matrix

    def action(self, board_state: np.ndarray) -> Move:

        x, y = self.pos

        valid_moves = []

        if board_state[x+1][y] == 0:
            v = self.value_matrix[x][y-1]
            valid_moves.append((Move.RIGHT, v))
        if board_state[x-1][y] == 0:
            v = self.value_matrix[x-2][y-1]
            valid_moves.append((Move.LEFT, v))
        if board_state[x][y+1] == 0:
            v = self.value_matrix[x-1][y]
            valid_moves.append((Move.UP, v))
        if board_state[x][y-1] == 0:
            v = self.value_matrix[x-1][y-2]
            valid_moves.append((Move.DOWN, v))

        if len(valid_moves) > 0:
            valid_moves = sorted(valid_moves, key=lambda x: x[1])
            rm = valid_moves[-1][0]
        else:
            rm = random.choice(list(Move))
        
        return rm


def play_game(n:int) -> np.ndarray:
    env = SnakeGame(n)
    a1_value = np.load('a1_avg_value.npy')
    a2_value = np.load('a2_avg_value.npy')
    ra1 = Random_Agent(a1_value, env.pos1, n)
    ra2 = Random_Agent(a2_value, env.pos2, n)

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
            break
    print(f'Got reward: {r}')
    return done 


if __name__ == '__main__':
    '''
    Just going to create a value table for the environment
    Then I create an agent that moves in the optimal direction
    '''

    # Make the environment 
    grid_size = 9 
    r_all = []

    for _ in range(int(1e1)):
        # This is deterministic, so the reward should be the same every time
        r = play_game(grid_size)
        r_all.append(r)
    print(r_all)