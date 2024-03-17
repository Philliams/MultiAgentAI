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

    def get_valid_moves(self, board_state, pos):
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

    def recursive_explore_moves(self, board_state, pos1, pos2, depth):

        valid_moves_1 = self.get_valid_moves(board_state, pos1)
        valid_moves_2 = self.get_valid_moves(board_state, pos2)

        average_advantage = {}

        if depth == 0:

            for v1 in valid_moves_1:

                average_advantage[v1] = 0

                for v2 in valid_moves_2:
                    x1, y1 = self.move_update(pos1, v1)
                    x2, y2 = self.move_update(pos2, v2)

                    if (x1, y1) == (x2, y2) or (x1, y1) == pos2:
                        # collision with each other
                        # this can be tuned to reduce/increase risk taking
                        average_advantage[v1] -= board_state.size / len(valid_moves_2)
                    elif (x2, y2) == pos1:
                        average_advantage[v1] += board_state.size / len(valid_moves_2)
                    else:
                        bfs1 = self.bfs((x1, y1), board_state)
                        bfs2 = self.bfs((x2, y2), board_state)

                        advantage = np.sum(bfs1 < bfs2)
                        disadvantage = np.sum(bfs1 > bfs2)
                        # squares we can reach first - squares they can reach first
                        average_advantage[v1] += (advantage - disadvantage) / len(valid_moves_2)

            # return best move and reward
            if len(average_advantage.keys()) > 0:
                optimal = max(average_advantage.items(), key=operator.itemgetter(1))
                return optimal
            else:
                return (Move.UP, -board_state.size)

        else:

            average_advantage = {}

            for v1 in valid_moves_1:

                average_advantage[v1] = 0

                for v2 in valid_moves_2:
                    x1, y1 = self.move_update(pos1, v1)
                    x2, y2 = self.move_update(pos2, v2)

                    if (x1 == x2) and (y1 == y2):
                        average_advantage[v1] -= board_state.size / len(valid_moves_2)
                    else:
                        board_state_ = board_state.copy()
                        board_state_[x1, y1] = -1
                        board_state_[x2, y2] = -1

                        _, reward = self.recursive_explore_moves(board_state_, (x1, y1), (x2, y2), depth - 1)
                        average_advantage[v1] += reward / len(valid_moves_2)

            if len(average_advantage.keys()) > 0:
                optimal = max(average_advantage.items(), key=operator.itemgetter(1))
                return optimal
            else:
                return (Move.UP, -board_state.size)

    
    def step(self, board_state, pos: tuple[int, int], other_pos: tuple[int, int]) -> Move:
        move, reward = self.recursive_explore_moves(board_state, pos, other_pos, self.n - 1)
        return move

    def bfs(self, pos, board_state):

        deltas = [
            (-1, 0), # left
            (1, 0), # right
            (0, 1), # up
            (0, -1) # down
        ]

        distances = np.zeros(board_state.shape)
        visited = np.full(board_state.shape, False)

        x, y = pos
        visited[x, y] = True

        queue = [pos]

        while queue:
            x, y = queue.pop(0)

            for dx, dy in deltas:
                xv = x + dx
                yv = y + dy

                # if not visited and not obstacle
                if (not visited[xv, yv]) and (board_state[xv, yv] == 0):
                    visited[xv, yv] = True
                    distances[xv, yv] = distances[x, y] + 1
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
            return Result.LOSE
        
        return False


    def reward_function(self):
        return (0,0)

    def step(self):

        m1 = self.agent1.step(self.state, self.pos1, self.pos2)
        m2 = self.agent2.step(self.state, self.pos2, self.pos1)

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

            # do some post_processing to align game coordinates with screen coordinates
            dataset.append(np.stack([
                np.flipud(r_grid.T),
                np.flipud(g_grid.T),
                np.flipud(b_grid.T)
            ], axis=0))

        write_gif(dataset, filepath, fps=15)


if __name__ == "__main__":
    agent1 = LazyAgent(id=1, start_move=Move.DOWN)
    agent2 = VoronoiAgent(id=2, forward_steps=1)

    game = SnakeGame(agent1, agent2, n=50)

    result = game.play()
    print(f"Player 1 : {result}")
    game.render_game("./data/dummy_lazy.gif")


