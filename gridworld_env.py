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
    """
    GridWorldEnv

    An environment for grid world navigation tasks.

    Attributes:
    metadata (dict): Metadata for the environment.
    layout (ndarray): The grid layout of the environment.
    grid_size_x (int): The size of the grid along the x-axis.
    grid_size_y (int): The size of the grid along the y-axis.
    action_space (gym.spaces.Discrete): The action space of the environment.
    observation_space (gym.spaces.Box): The observation space of the environment.
    state (tuple): The current state of the agent.
    success_prob (float): The probability of successfully taking an action.
    window_size (tuple): The size of the rendering window.
    screen (pygame.Surface): The Pygame screen object for rendering.
    clock (pygame.time.Clock): The Pygame clock object for rendering.

    Methods:
    __init__(layout_file, success_prob=0.8, rewards_map=None)
    _load_layout(filename)
    get_reward(state)
    get_type(state)
    find_positions(cell_type)
    get_terminal_states()
    reset()
    step(action)
    is_valid_state(state)
    get_transitions(state, action)
    _action_to_move(action_enum)
    _random_side_move(action_enum)
    render(mode='human')
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, layout_file, success_prob=0.8, rewards_map=None):
        super(GridWorldEnv, self).__init__()

        # Define default rewards for each cell type
        default_rewards_map = {
            CellType.TARGET: 0.0,
            CellType.START: -0.1,
            CellType.OBSTACLE: 0.0,
            CellType.TREASURE: 5.0,
            CellType.TRAP: -20.0,
            CellType.EMPTY: -0.1
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
    
    def is_valid_state(self, state):
        """
        Check if a state (x, y) is a valid state in the grid.

        Parameters:
        x (int): The x-coordinate.
        y (int): The y-coordinate.

        Returns:
        bool: True if the state is valid, False otherwise.
        """
        if 0 <= state[0] < self.grid_size_x and 0 <= state[1] < self.grid_size_y:
            return self.layout[state[0], state[1]] != CellType.OBSTACLE
        return False
    
    def get_transitions(self, state, action):
        """
        Get the possible transitions for a given state and action for MDP estimation.

        Parameters:
        state (tuple): The current state (x, y).
        action (Action): The action to be performed.

        Returns:
        List of tuples: Each tuple contains (next_state, probability).
        """
        transitions = []
        x, y = state

        # Define possible moves
        moves = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1)
        }

        # Calculate the next state for the action
        move = moves.get(action)
        new_x, new_y = x + move[0], y + move[1]

        # Handle the primary action
        if self.is_valid_state((new_x, new_y)):
            next_state = (new_x, new_y)
        else:
            # If the move is not valid, the agent stays in the same state
            next_state = state
        
        # Add the primary transition with success probability
        transitions.append((next_state, self.success_prob))

        # Handle slipping probabilities
        slip_prob = (1.0 - self.success_prob) / 2
        for slip_move in [(move[1], move[0]), (-move[1], -move[0])]:  # Perpendicular moves
            new_x, new_y = x + slip_move[0], y + slip_move[1]
            if self.is_valid_state((new_x, new_y)):
                next_state = (new_x, new_y)
            else:
                # If the slip move is not valid, the agent stays in the same state
                next_state = state
            transitions.append((next_state, slip_prob))

        return transitions

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
            return random.choice([(0, -1), (0, -1)])
        else:
            return random.choice([(-1, 0), (1, 0)])

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
