import numpy as np
from enum import Enum
import random
from array2gif import write_gif
import operator

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

    def step(self, board_state: np.ndarray, pos: tuple[int, int], other_pos: tuple[int, int]) -> Move:
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
        
class LazyAgent:
    def __init__(self, id:int, start_move: Move):
        self.id = id
        self.last_move = start_move
    
    def step(self, board_state, pos: tuple[int, int], other_pos: tuple[int, int]) -> Move:
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

        if self.last_move in valid_moves:
            return self.last_move
        
        elif len(valid_moves) > 0:
            self.last_move = random.choice(valid_moves)
            return self.last_move
        else:
            return random.choice(list(Move))

class VoronoiAgent:

    def __init__(self, id:int, forward_steps:int = 1):
        self.n = forward_steps
        self.id = id
        self.n = forward_steps - 1
        self.iter = 0

    def get_valid_moves(self, pos, board_state):
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
        return valid_moves
    
    def get_move_update(self, pos, move):
        x, y = pos
        if move == Move.UP:
            return (x, y+1)
        if move == Move.DOWN:
            return (x, y-1)
        if move == Move.LEFT:
            return (x-1, y)
        if move == Move.RIGHT:
            return (x+1, y)

    def step(self, board_state, pos: tuple[int, int], other_pos: tuple[int, int]):
        return self.recursive_step(board_state, pos, other_pos, self.n)[0]

    def recursive_step(self, board_state, pos, other_pos, depth) -> Move:
        
        average_rewards = {}

        valid1 = self.get_valid_moves(pos, board_state)
        valid2 = self.get_valid_moves(other_pos, board_state)

        for v1 in valid1:

            average_rewards[v1] = 0

            for v2 in valid2:

                x1, y1 = self.get_move_update(pos, v1)
                x2, y2 = self.get_move_update(other_pos, v2)
                board_state_ = board_state.copy()

                if depth == 0:

                    bfs1 = self.bfs((x1, y1), board_state_.copy())
                    bfs2 = self.bfs((x2, y2), board_state_.copy())



                    r = (
                        np.count_nonzero(bfs1)
                        # - np.count_nonzero(bfs2)
                        + np.sum(bfs1 < bfs2)
                        # - np.sum(bfs1 > bfs2)
                    )
                    average_rewards[v1] += r / len(valid2)

                else:
                    board_state_[x1, y1] = self.id
                    board_state_[x2, y2] = self.id
                    _, r = self.recursive_step(board_state_, (x1, y1), (x2, y2), depth - 1)
                    average_rewards[v1] += r / len(valid2)
        
        if len(average_rewards.keys()) > 0:
            optimal = max(average_rewards.items(), key=operator.itemgetter(1))
            return optimal
        else:
            return (Move.UP, -board_state.size)

    def bfs(self, pos, board_state):
        board_state = board_state.copy()
        visited = np.full(board_state.shape, False)
        distances = np.zeros(board_state.shape)

        x, y = pos
        visited[x, y] = True
        board_state[x, y] = self.id
        queue = [pos]

        deltas = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0)
        ]

        while queue:
            x1, y1 = queue.pop(0)

            for dx, dy in deltas:
                xv = x1 + dx
                yv = y1 + dy
                if (not visited[xv, yv]) and (board_state[xv, yv] == 0):
                    distances[xv, yv] = distances[x1, y1] + 1
                    visited[xv, yv] = True
                    queue.append((xv, yv))

        return distances


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
            return Result.WIN
        
        return False


    def reward_function(self):
        return (0,0)

    def step(self):

        x1, y1 = self.pos1
        x2, y2 = self.pos2

        self.state[x1, y1] = self.agent1.id
        self.state[x2, y2] = self.agent2.id

        m1 = self.agent1.step(self.state, self.pos1, self.pos2)
        m2 = self.agent2.step(self.state, self.pos2, self.pos1)

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

            # do some post_processing to align game coordinates with screen coordinates
            dataset.append(np.stack([
                np.flipud(r_grid.T),
                np.flipud(g_grid.T),
                np.flipud(b_grid.T)
            ], axis=0))

        write_gif(dataset, filepath, fps=15)


if __name__ == "__main__":
    # agent1 = LazyAgent(id=1, start_move=Move.DOWN)
    agent1 = VoronoiAgent(id=1, forward_steps=1)
    agent2 = VoronoiAgent(id=2, forward_steps=2)

    game = SnakeGame(agent1, agent2, n=50)

    result = game.play()
    print(f"Player 1 : {result}")
    game.render_game("./data/dummy_lazy.gif")


