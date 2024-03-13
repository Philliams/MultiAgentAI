import numpy as np
from enum import Enum
import random
from array2gif import write_gif

class Move(Enum):

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Result(Enum):

    WIN = 1
    TIE = 0
    LOSE = -1
    
class AvoidAgent:
    def __init__(self, id:int):
        self.id = id

    def step(self, board_state: np.ndarray, pos: tuple[int, int]) -> Move:
        x, y = pos

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
            return random.choice(valid_moves)
        else:
            return random.choice(list(Move))

class SnakeGame:

    def __init__(self, agent1, agent2, n=9):
        self.agent1 = agent1
        self.agent2 = agent2

        self.state = -np.ones((n+2, n+2))
        self.state[1:n+1, 1:n+1] = np.zeros((n, n))
        
        self.history = [self.state.copy()]

        self.pos1 = (n//4 + 1, n//2 + 1)
        self.pos2 = (1 + (3 * n // 4), n//2 + 1)

        self.add_frame()

    def terminated(self):
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
        
        return False


    def reward_function(self):
        return (0,0)

    def step(self):

        m1 = self.agent1.step(self.state, self.pos1)
        m2 = self.agent2.step(self.state, self.pos2)

        x1, y1 = self.pos1
        x2, y2 = self.pos2

        self.state[x1, y1] = self.agent1.id
        self.state[x2, y2] = self.agent2.id

        self.pos1 = self.move_update(self.pos1, m1)
        self.pos2 = self.move_update(self.pos2, m2)

        self.add_frame()

        done = self.terminated()
        reward = self.reward_function()

        # Assumes full observability of the environment
        s_prime = self.state.copy()

        return done,reward,s_prime


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

    def render_game(self, filepath:str):

        dataset = []

        for state in self.history:
            r_grid = state.copy()
            g_grid = state.copy()
            b_grid = state.copy()

            # set wall color
            r_grid[state == -1] = 255
            g_grid[state == -1] = 255
            b_grid[state == -1] = 255

            # set agent 1 color
            r_grid[state == self.agent1.id] = 255
            g_grid[state == self.agent1.id] = 0
            b_grid[state == self.agent1.id] = 0

            # set agent 2 color
            r_grid[state == self.agent2.id] = 0
            g_grid[state == self.agent2.id] = 0
            b_grid[state == self.agent2.id] = 255

            dataset.append(np.stack([r_grid, g_grid, b_grid], axis=0))

        write_gif(dataset, filepath, fps=15)




if __name__ == "__main__":
    agent1 = AvoidAgent(id=1)
    agent2 = AvoidAgent(id=2)

    game = SnakeGame(agent1, agent2, n=50)

    game.play()
    game.render_game("./data/dummy.gif")


