'''
Random Agets to explore the environment? Create Value table???
'''

import numpy as np
import random

from game import *


class Random_Agent():
    def __init__(self, starting_pos:tuple[int,int], n:int=9):
        self.value_table = np.array([[0 for _ in range(n)] for _ in range(n)]) 
        self.count_table = np.array([[0 for _ in range(n)] for _ in range(n)])
        self.pos = starting_pos
        self.prev_pos = []

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
            valid_moves.append(Move.RIGHT)
        if board_state[x-1][y] == 0:
            valid_moves.append(Move.LEFT) 
        if board_state[x][y+1] == 0:
            valid_moves.append(Move.UP)
        if board_state[x][y-1] == 0:
            valid_moves.append(Move.DOWN) 

        if len(valid_moves) > 0:
            rm = random.choice(valid_moves)
        else:
            rm = random.choice(list(Move))
        
        return rm


if __name__ == '__main__':
    '''
    Just going to create a value table for the environment
    Then I create an agent that moves in the optimal direction
    '''

    # Make the environment 
    grid_size = 5 
    env = SnakeGame(n=grid_size)

    # Create the agents
    ra1 = Random_Agent(env.pos1, grid_size)
    ra2 = Random_Agent(env.pos2, grid_size)

    # Play the game
    c = 0
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

        c += 1
        if done or c == 4:
            print('Rewards for player 1')
            print(ra1.value_table)
            print(ra1.count_table)
            break

    # Print the value table 
    