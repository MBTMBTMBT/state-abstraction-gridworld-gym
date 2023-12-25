import gym
from gym import spaces
import numpy as np
import random
import pygame
from enum import Enum


class Action(Enum):
    """Enumeration for possible actions in the GridWorld environment."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellType(Enum):
    """Enumeration for different types of cells in the GridWorld layout."""
    TARGET = 'T'
    START = 'S'
    OBSTACLE = 'O'
    TREASURE = '+'
    TRAP = 'X'
    EMPTY = '.'


class GridWorldEnv(gym.Env):
    """A custom environment for GridWorld compatible with OpenAI Gym."""

    metadata = {'render.modes': ['human']}

    def __init__(self, layout_file, success_prob=0.8, rewards_map=None):
        """
        Initialize the GridWorld environment.
        
        Parameters:
        layout_file (str): The file path for the grid layout.
        success_prob (float): Probability of successful action movement.
        rewards_map (dict, optional): Custom rewards map for different cell types.
        """
        super(GridWorldEnv, self).__init__()

        # Define default rewards for each cell type
        default_rewards_map = {
            CellType.TARGET: 0.0,
            CellType.START: -1.0,
            CellType.OBSTACLE: -1.0,
            CellType.TREASURE: 5.0,
            CellType.TRAP: -20.0,
            CellType.EMPTY: -1.0
        }

        # Use the provided rewards map or default if none is provided
        self.rewards_map = rewards_map if rewards_map else default_rewards_map
        # Load the layout from the specified file
        self.layout = self._load_layout(layout_file)
        # Determine grid size for defining spaces
        self.grid_size_x, self.grid_size_y = self.layout.shape

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=0, high=max(self.grid_size_x, self.grid_size_y), 
                                            shape=(2,), dtype=np.int32)

        # Initialize state and success probability
        self.state = None
        self.success_prob = success_prob

        # Setup for rendering with Pygame
        self.window_size = (self.grid_size_y * 50, self.grid_size_x * 50)
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.clock = pygame.time.Clock()

    def _load_layout(self, filename):
        """
        Load the grid layout from a file, converting each cell to a CellType enum.
        Parameters:
        filename (str): The file path for the grid layout.
        """
        with open(filename, 'r') as f:
            layout = [list(line.strip()) for line in f.readlines()]
        # Convert string representations to CellType enums
        return np.array([[CellType(cell) for cell in row] for row in layout])

    def get_reward(self, state):
        """
        Get the reward for a given state.
        Parameters:
        state (tuple): The state to retrieve the reward for.
        """
        cell_type_enum = self.layout[state[0], state[1]]
        return self.rewards_map[cell_type_enum]
    
    def get_type(self, state):
        """
        Get the type of a cell in a given state.
        Parameters:
        state (tuple): The state to get the cell type for.
        """
        return self.layout[state[0], state[1]]

    def find_positions(self, cell_type):
        """
        Find all positions of a given cell type in the grid.
        Parameters:
        cell_type (CellType): The cell type to find positions for.
        """
        return [(i, j) for i in range(self.grid_size_x) for j in range(self.grid_size_y) if self.layout[i, j] == cell_type]

    def get_terminal_states(self):
        """Get all states that are considered terminal (end of episode)."""
        return self.find_positions(CellType.TARGET)

    def reset(self):
        """Reset the environment to the initial state and return the starting state."""
        start_positions = self.find_positions(CellType.START)
        if not start_positions:
            raise ValueError("No starting position in layout")
        self.state = start_positions[0]
        return np.array(self.state)

    def step(self, action):
        """
        Perform an action in the environment and return the result.
        Parameters:
        action (int): The action to be performed.
        """
        action_enum = Action(action)
        assert self.action_space.contains(action), f"{action_enum} is an invalid action"

        # Determine movement based on action and success probability
        if random.random() < self.success_prob:
            move = self._action_to_move(action_enum)
        else:
            move = self._random_side_move(action_enum)

        # Calculate new state and check if it's valid
        new_state = (self.state[0] + move[0], self.state[1] + move[1])
        if 0 <= new_state[0] < self.grid_size_x and 0 <= new_state[1] < self.grid_size_y and \
           self.layout[new_state[0], new_state[1]] != CellType.OBSTACLE:
            self.state = new_state

        # Determine reward and if the episode is done
        reward = self.get_reward(self.state)
        done = self.layout[self.state[0], self.state[1]] == CellType.TARGET

        return np.array(self.state), reward, done, {}

    def _action_to_move(self, action_enum):
        """
        Map an action enum to a movement (delta x, delta y).
        Parameters:
        action_enum (Action): The action enum to be converted to a move.
        """
        action_mapping = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }
        return action_mapping[action_enum]

    def _random_side_move(self, action_enum):
        """
        Determine a random side move for the slipping action.
        Parameters:
        action_enum (Action): The action enum to determine the side move for.
        """
        if action_enum in [Action.UP, Action.DOWN]:
            return random.choice([(-1, 0), (1, 0)])
        else:
            return random.choice([(0, -1), (0, 1)])

    def render(self, mode='human'):
        """
        Render the environment's current state.
        Parameters:
        mode (str): The mode to render in ('human').
        """
        if mode == 'human':
            colors = {
                CellType.EMPTY: (192, 192, 192),
                CellType.TARGET: (255, 215, 0),
                CellType.START: (0, 0, 255),
                CellType.OBSTACLE: (0, 0, 0),
                CellType.TREASURE: (0, 255, 0),
                CellType.TRAP: (255, 0, 0)
            }
            # Render each cell with the appropriate color
            for i in range(self.grid_size_x):
                for j in range(self.grid_size_y):
                    cell_type = self.layout[i, j]
                    color = colors[cell_type]
                    rect = pygame.Rect(j * 50, i * 50, 50, 50)
                    pygame.draw.rect(self.screen, color, rect)

            # Render the agent's current position
            agent_color = (255, 255, 255)
            agent_rect = pygame.Rect(self.state[1] * 50, self.state[0] * 50, 50, 50)
            pygame.draw.rect(self.screen, agent_color, agent_rect)

            pygame.display.flip()
            self.clock.tick(60)


# Main function to run the environment
if __name__ == '__main__':
    env = GridWorldEnv('layout.txt')
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()

    pygame.quit()
