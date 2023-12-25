import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pygame


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, layout_file, success_prob=0.8):
        super(GridWorldEnv, self).__init__()

        self.rewards_map = {
            'T': 0.0,  # Target
            'S': -1.0, # Starting point
            'O': -1.0, # Obstacle
            '+': 5.0,  # Treasure
            'X': -20.0,# Trap
            '.': -1.0  # Empty cell
        }
        self.layout = self._load_layout(layout_file)
        self.grid_size_x, self.grid_size_y = self.layout.shape

        self.action_space = spaces.Discrete(4) # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=max(self.grid_size_x, self.grid_size_y), 
                                            shape=(2,), dtype=np.int32)

        self.state = None
        self.success_prob = success_prob  # Probability of successful movement

        self.window_size = (self.grid_size_y * 50, self.grid_size_x * 50)  # Window size in pixels
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

    def _load_layout(self, filename):
        with open(filename, 'r') as f:
            layout = [list(line.strip()) for line in f.readlines()]
        return np.array(layout)
    
    def get_reward(self, state):
        cell_type = self.layout[state[0], state[1]]
        return self.rewards_map[cell_type]

    def get_type(self, state):
        return self.layout[state[0], state[1]]

    def find_positions(self, cell_type):
        return [(i, j) for i in range(self.grid_size_x) for j in range(self.grid_size_y) if self.layout[i, j] == cell_type]

    def get_terminal_states(self):
        return self.find_positions('T')

    def reset(self):
        start_positions = self.find_positions('S')
        if not start_positions:
            raise ValueError("No starting position in layout")
        self.state = start_positions[0]  # Choose the first start position
        return np.array(self.state)

    def step(self, action):
        assert self.action_space.contains(action), f"{action} is an invalid action"

        # Determine if the action is successful or slips
        if random.random() < self.success_prob:
            # Successful action
            move = self._action_to_move(action)
        else:
            # Slip to a random side
            move = self._random_side_move(action)

        new_state = (self.state[0] + move[0], self.state[1] + move[1])

        # Check boundaries and obstacles
        if 0 <= new_state[0] < self.grid_size_x and 0 <= new_state[1] < self.grid_size_y and \
           self.layout[new_state[0], new_state[1]] != 'O':
            self.state = new_state

        reward = self.get_reward(self.state)
        done = self.layout[self.state[0], self.state[1]] == 'T'

        return np.array(self.state), reward, done, {}

    def _action_to_move(self, action):
        action_mapping = {
            0: (-1, 0), # up
            1: (1, 0),  # down
            2: (0, -1), # left
            3: (0, 1)   # right
        }
        return action_mapping[action]

    def _random_side_move(self, action):
        # Determine side moves based on the action
        if action in [0, 1]:  # Up or down
            return random.choice([(-1, 0), (1, 0)])
        else:  # Left or right
            return random.choice([(0, -1), (0, 1)])

    def render(self, mode='human'):
        if mode == 'human':
            # Define colors for each cell type
            colors = {
                '.': (192, 192, 192),  # Empty cell
                'T': (255, 215, 0),    # Target
                'S': (0, 0, 255),      # Starting point
                'O': (0, 0, 0),        # Obstacle
                '+': (0, 255, 0),      # Treasure
                'X': (255, 0, 0)       # Trap
            }

            # Fill each cell with the corresponding color
            for i in range(self.grid_size_x):
                for j in range(self.grid_size_y):
                    color = colors[self.layout[i, j]]
                    rect = pygame.Rect(j * 50, i * 50, 50, 50)
                    pygame.draw.rect(self.screen, color, rect)

            # Mark the agent's position
            agent_color = (255, 255, 255)  # White
            agent_rect = pygame.Rect(self.state[1] * 50, self.state[0] * 50, 50, 50)
            pygame.draw.rect(self.screen, agent_color, agent_rect)

            pygame.display.flip()  # Update the full display
            self.clock.tick(60)    # Limit the frame rate


if __name__ == '__main__':
    env = GridWorldEnv('layout.txt')
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()

    pygame.quit()
