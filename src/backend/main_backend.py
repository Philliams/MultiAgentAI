import numpy as np
from enum import Enum
import random
import math
from array2gif import write_gif
import operator
import time
from matplotlib import pyplot as plt
import pandas as pd

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

class EpsilonGreedyAgent:

    def __init__(self, agent, epsilon):
        self.e = epsilon
        self.agent = agent
        self.id = self.agent.id

    def step(self, board_state, pos: tuple[int, int], other_pos: tuple[int, int]):
        if np.random.uniform() >= self.e:
            return self.agent.step(board_state, pos, other_pos)
        else:
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


class TabularAgent:

    def __init__(self, n, gamma, id):
        # one feature for each cell, 2 features per position and a fixed bias feature
        self.weights = np.random.uniform(size=(n*n + 5,))
        self.n = n
        self.gamma = gamma
        self.alpha = 0.01
        self.history = []
        self.id=id
        self.e = 0.8

    def reset_history(self):
        self.history = []

    def update_weights(self, reward):
        k = len(self.history)
        for i, timestep in enumerate(self.history):
            _, _, features = timestep
            discounted_reward = reward * math.pow(self.gamma, k - i)
            v = np.dot(features, self.weights)
            self.weights += self.alpha * (discounted_reward - v) * features

        self.reset_history()

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
        
    def get_feature_vector(self, board, pos, other_pos):
        arr = board[1:-1, 1:-1].reshape(self.n * self.n).copy()
        arr[arr != 0] = 1
        positional_features = np.array([1, pos[0] / self.n, pos[1] / self.n, other_pos[0] / self.n, other_pos[1] / self.n])

        return np.concatenate([arr, positional_features])


    def step(self, board_state, pos, other_pos):

        features = self.get_feature_vector(board_state, pos, other_pos)
        self.history.append((pos, other_pos, features))

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

        max_ = None

        for move in list(Move):
            x1, y1 = self.get_move_update(pos, move)
            x2, y2 = other_pos
            new_state = board_state.copy()
            new_state[x1, y1] = 1

            features = self.get_feature_vector(board_state, (x1, y1), (x2, y2))
            v = np.dot(features, self.weights)
            if (max_ is None) or (v > max_[1]):
                max_ = (move, v)

        if np.random.uniform() < self.e:
            return max_[0]
        elif len(valid_moves) > 0:
            return random.choice(valid_moves)
        else:
            return Move.UP



class SnakeGame:

    def __init__(self, agent1, agent2, n=9):
        self.agent1 = agent1
        self.agent2 = agent2
        self.n = n
        self.reset()
        self.__dummy_agent = VoronoiAgent(2, 0)
        self.voronoi_ratios = []
        self.k = 0
        self.compute_voronoi = False

    def reset(self):

        self.state = -np.ones((self.n+2, self.n+2))
        self.state[1:self.n+1, 1:self.n+1] = np.zeros((self.n, self.n))
        
        self.history = [self.state.copy()]

        self.pos1 = (self.n//4 + 1, self.n//2 + 1)
        self.pos2 = (1 + (3 * self.n // 4), self.n//2 + 1)

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

        if self.compute_voronoi:
            bfs1 = self.__dummy_agent.bfs(self.pos1, self.state.copy())
            bfs2 = self.__dummy_agent.bfs(self.pos2, self.state.copy())

            interacting = ((bfs1 != 0) & (bfs2 != 0)).any()

            dummy_state1 = self.state.copy()
            dummy_state1[bfs1 < bfs2] = 0
            dummy_state1[bfs1 >= bfs2] = 1
            dummy_state1[self.state != 0] = 1
            paths = compute_paths(dummy_state1, (3, 3))
            try:
                hamiltonian1 = max(paths, key=lambda l: len(l))
            except:
                hamiltonian1 = []

            dummy_state2 = self.state.copy()
            dummy_state2[bfs1 > bfs2] = 0
            dummy_state2[bfs1 <= bfs2] = 1
            dummy_state2[self.state != 0] = 1
            paths = compute_paths(dummy_state2, (3, 3))
            try:
                hamiltonian2 = max(paths, key=lambda l: len(l))
            except:
                hamiltonian2 = []

            self.voronoi_ratios.append({
                'k': self.k,
                '1': (np.sum(dummy_state1 == 0), len(hamiltonian1)),
                '2': (np.sum(dummy_state2 == 0), len(hamiltonian2)),
                'interacting' : interacting
            })

            self.k += 1

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


def compute_paths(G: np.array, pos, seen=None,path=None):
    if seen is None: seen = []
    if path is None: path = [pos]

    seen.append(pos)
    paths = []
    deltas = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1)
    ]

    x, y = pos

    for dx, dy in deltas:
        t = (x + dx, y + dy)
        if (t not in seen) and (G[t[0], t[1]] == 0):
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(compute_paths(G, t, seen[:], t_path))
    return paths

def generate_plot(data):
    data.boxplot(column=['hamiltonian length'], by='voronoi cell size')

    cell_sizes = sorted(data['voronoi cell size'].unique().tolist())

    plt.title('')
    plt.suptitle('')
    plt.ylabel("Longest Hamiltonian Length")
    plt.plot(cell_sizes, cell_sizes, color='red')
    plt.savefig(f'./data/voronoi_hamiltonian_boxplot.png', bbox_inches='tight')

def generate_plot_game_scores(data):
    
    data.boxplot(column=['score'], by='opponent')
    plt.title('')
    plt.suptitle('')
    plt.ylabel("Win Percentage(%)")
    plt.savefig(f'./data/win_percentage_boxplot.png', bbox_inches='tight')

    data.boxplot(column=['game length'], by='opponent')
    plt.title('')
    plt.suptitle('')
    plt.ylabel("Game Length")
    plt.savefig(f'./data/game_length_boxplot.png', bbox_inches='tight')
    

if __name__ == "__main__":

    scores = []
    ratios = []
    num_repeats = 1
    compute_voronoi = False

    opponents = [
        ("Random Walk", lambda: AvoidAgent(id=2)),
        ("Lazy (ε = 0.0)", lambda: EpsilonGreedyAgent(LazyAgent(id=2, start_move=Move.DOWN), 0.0)),
        ("Lazy (ε = 0.2)", lambda: EpsilonGreedyAgent(LazyAgent(id=2, start_move=Move.DOWN), 0.2)),
        ("Lazy (ε = 0.4)", lambda: EpsilonGreedyAgent(LazyAgent(id=2, start_move=Move.DOWN), 0.4)),
        ("Lazy (ε = 0.6)", lambda: EpsilonGreedyAgent(LazyAgent(id=2, start_move=Move.DOWN), 0.6)),
        ("Lazy (ε = 0.8)", lambda: EpsilonGreedyAgent(LazyAgent(id=2, start_move=Move.DOWN), 0.8)),
        ("Voronoi (ε = 0.0)", lambda: EpsilonGreedyAgent(VoronoiAgent(id=2, forward_steps=1), 0.0)),
        ("Voronoi (ε = 0.2)", lambda: EpsilonGreedyAgent(VoronoiAgent(id=2, forward_steps=1), 0.2)),
        ("Voronoi (ε = 0.4)", lambda: EpsilonGreedyAgent(VoronoiAgent(id=2, forward_steps=1), 0.4)),
        ("Voronoi (ε = 0.6)", lambda: EpsilonGreedyAgent(VoronoiAgent(id=2, forward_steps=1), 0.6)),
        ("Voronoi (ε = 0.8)", lambda: EpsilonGreedyAgent(VoronoiAgent(id=2, forward_steps=1), 0.8)),
    ]

    for opp, a2 in opponents:
        for i in range(num_repeats):

            agent1 = VoronoiAgent(id=1, forward_steps=1)
            agent2 = a2()

            game = SnakeGame(agent1, agent2, n=9)
            game.compute_voronoi = compute_voronoi
            result = game.play()
            scores.append({
                "score" : 1 if result == Result.WIN else 0,
                "ties" : 1 if result == Result.TIE else 0,
                "game length" : len(game.history),
                "opponent" : opp
            })

            if compute_voronoi:
                for d in game.voronoi_ratios:
                    bfs1, h1 = d['1']
                    bfs2, h2 = d['2']

                    if h1 > 0 and bfs1 > 0:
                        ratios.append({'hamiltonian length' : h1, 'voronoi cell size' : bfs1})

                    if h2 > 0 and bfs2 > 0:
                        ratios.append({'hamiltonian length' : h2, 'voronoi cell size' : bfs2})

    # generate a sample game gif
    agent1 = VoronoiAgent(id=1, forward_steps=1)
    agent2 = EpsilonGreedyAgent(VoronoiAgent(id=2, forward_steps=1), epsilon=0.1)
    game = SnakeGame(agent1, agent2, n=50)
    result = game.play()
    game.render_game("./data/game_viz.gif")

    # print win statistics
    df = pd.DataFrame(scores)
    print(df.groupby("opponent").agg(["mean", "std"]))

    # use this to generate win percentage plots and voronoi plots
    generate_plot_game_scores(pd.DataFrame(scores))
    if compute_voronoi:
        generate_plot(pd.DataFrame(ratios))

    

    