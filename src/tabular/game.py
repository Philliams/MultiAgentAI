import numpy as np
from enum import Enum
import random

class Move(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class Result(Enum):
    WIN = 1
    TIE = 0
    LOSE = -1


class SnakeGame:

    def __init__(self, n=9):
        '''
        Want an openAI like gym environment, AKA a game that can be played by 2 players
        '''
        self.state = -np.ones((n+2, n+2))
        self.state[1:n+1, 1:n+1] = np.zeros((n, n))
        
        self.history = [self.state.copy()]

        # Copied
        self.pos1 = (n//4 + 1, n//2 + 1)
        self.pos2 = (1 + (3 * n // 4), n//2 + 1)

        self.add_frame()

        self.reward = 0
        self.max_steps = int(1e7)
        self.steps = 0

    def terminated(self):
        '''
        Need a reward?
        '''
        x1, y1 = self.pos1
        x2, y2 = self.pos2

        if (x1 == x2) and (y1 == y2):
            return Result.TIE
        elif self.state[x1, y1] != 0 and self.state[x2, y2] != 0:
            return Result.TIE
        elif self.state[x1, y1] != 0:
            return Result.LOSE
        elif self.state[x2, y2] != 0:
            return Result.LOSE

        if self.steps == self.max_steps:
            return Result.TIE 

        return False

    def step(self, m1:Move, m2:Move):
        self.steps += 1
        x1, y1 = self.pos1
        x2, y2 = self.pos2

        self.state[x1, y1] = 1  # player 1
        self.state[x2, y2] = 2  # player 2

        self.pos1 = self.move_update(self.pos1, m1)
        self.pos2 = self.move_update(self.pos2, m2)

        self.add_frame()

        done = self.terminated()
        self.reward += 1

        # Assumes full observability of the environment
        s_prime = self.state.copy()

        # OpenAI Gym Return: https://gymnasium.farama.org
        #return s_prime, self.reward, done
        return ((self.pos1, self.pos2), self.reward, done)


    def play(self, max_steps = 100000000):

        done = False

        while not done:

            max_steps -= 1
            if max_steps <= 0:
                raise Exception("Game Failed to terminate in max steps.")

            done,_,_= self.step()
            
        return done

    def move_update(self, pos:tuple[int, int], move:Move) -> tuple[int, int]:
        x, y = pos
        if move == Move.UP:
            return (x, y+1)
        if move == Move.DOWN:
            return (x, y-1)
        if move == Move.LEFT:
            return (x-1, y)
        if move == Move.RIGHT:
            return (x+1, y)
        
    def add_frame(self):

        frame = self.state.copy()
        x1, y1 = self.pos1
        x2, y2 = self.pos2

        frame[x1, y1] = -1
        frame[x2, y2] = -1

        self.history.append(frame)
