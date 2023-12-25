import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygame
from enum import Enum


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellType(Enum):
    TARGET = 'T'
    START = 'S'
    OBSTACLE = 'O'
    TREASURE = '+'
    TRAP = 'X'
    EMPTY = '.'


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, layout_file, success_prob=0.8, rewards_map=None):
        super(GridWorldEnv, self).__init__()

        # Default rewards map
        default_rewards_map = {
            CellType.TARGET: 0.0,
            CellType.START: -1.0,
            CellType.OBSTACLE: -1.0,
            CellType.TREASURE: 5.0,
            CellType.TRAP: -20.0,
            CellType.EMPTY: -1.0
        }

        self.rewards_map = rewards_map if rewards_map else default_rewards_map
        self.layout = self._load_layout(layout_file)
        self.grid_size_x, self.grid_size_y = self.layout.shape

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=0, high=max(self.grid_size_x, self.grid_size_y), 
                                            shape=(2,), dtype=np.int32)

        self.state = None
        self.success_prob = success_prob

        # Pygame setup
        self.window_size = (self.grid_size_y * 50, self.grid_size_x * 50)
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

    def _load_layout(self, filename):
        # [Your existing _load_layout method, modified to use CellType enums]
        with open(filename, 'r') as f:
            layout = [list(line.strip()) for line in f.readlines()]
        return np.array(layout, dtype=CellType)
    
    def get_reward(self, state):
        cell_type_str = self.layout[state[0], state[1]]
        cell_type_enum = CellType(cell_type_str)
        return self.rewards_map[cell_type_enum]

    def get_type(self, state):
        return self.layout[state[0], state[1]]

    def find_positions(self, cell_type):
        return [(i, j) for i in range(self.grid_size_x) for j in range(self.grid_size_y) if self.layout[i, j] == cell_type]

    def get_terminal_states(self):
        return self.find_positions(CellType.TARGET.value)

    def reset(self):
        start_positions = self.find_positions(CellType.START.value)
        if not start_positions:
            raise ValueError("No starting position in layout")
        self.state = start_positions[0]  # Choose the first start position
        return np.array(self.state)

    def step(self, action):
        action_enum = Action(action)
        assert self.action_space.contains(action), f"{action_enum} is an invalid action"

        # Determine if the action is successful or slips
        if random.random() < self.success_prob:
            move = self._action_to_move(action_enum)
        else:
            move = self._random_side_move(action_enum)

        new_state = (self.state[0] + move[0], self.state[1] + move[1])

        # Check boundaries and obstacles
        if 0 <= new_state[0] < self.grid_size_x and 0 <= new_state[1] < self.grid_size_y and \
        self.layout[new_state[0], new_state[1]] != CellType.OBSTACLE.value:
            self.state = new_state

        reward = self.get_reward(self.state)
        done = self.layout[self.state[0], self.state[1]] == CellType.TARGET.value

        return np.array(self.state), reward, done, {}

    def _action_to_move(self, action_enum):
        action_mapping = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        return action_mapping[action_enum]

    def _random_side_move(self, action_enum):
        if action_enum in [Action.UP, Action.DOWN]:
            return random.choice([(-1, 0), (1, 0)])
        else:
            return random.choice([(0, -1), (0, 1)])

    def render(self, mode='human'):
        if mode == 'human':
            colors = {
                CellType.EMPTY.value: (192, 192, 192),
                CellType.TARGET.value: (255, 215, 0),
                CellType.START.value: (0, 0, 255),
                CellType.OBSTACLE.value: (0, 0, 0),
                CellType.TREASURE.value: (0, 255, 0),
                CellType.TRAP.value: (255, 0, 0)
            }

            for i in range(self.grid_size_x):
                for j in range(self.grid_size_y):
                    color = colors[self.layout[i, j]]
                    rect = pygame.Rect(j * 50, i * 50, 50, 50)
                    pygame.draw.rect(self.screen, color, rect)

            agent_color = (255, 255, 255)  # White
            agent_rect = pygame.Rect(self.state[1] * 50, self.state[0] * 50, 50, 50)
            pygame.draw.rect(self.screen, agent_color, agent_rect)

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == '__main__':
    env = GridWorldEnv('layout.txt')
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()

    pygame.quit()
