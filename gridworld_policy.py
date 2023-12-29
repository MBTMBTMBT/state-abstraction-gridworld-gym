from gridworld_env import GridWorldEnv, Action, CellType
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math


class SingleStatePolicy:
    """
    The SingleStatePolicy class represents a policy for a single state in a Markov Decision Process (MDP).

    Attributes:
    - up (float): The probability of taking the "up" action.
    - down (float): The probability of taking the "down" action.
    - left (float): The probability of taking the "left" action.
    - right (float): The probability of taking the "right" action.

    Methods:
    - to_list(): Converts the policy to a list of action probabilities.
    - update_from_list(values): Updates the policy from a list of action probabilities.
    - __str__(): Returns a string representation of the policy.
    - kl_divergence(policy_at_state_p, policy_at_state_q): Computes the Kullback-Leibler divergence between two policies.

    """
    def __init__(self, up=0.0, down=0.0, left=0.0, right=0.0) -> None:
        self.up = up
        self.down = down
        self.left = left
        self.right = right

    def to_list(self):
        """Convert the policy to a list of action probabilities."""
        return [self.up, self.down, self.left, self.right]

    def update_from_list(self, values):
        """Update the policy from a list of action probabilities."""
        self.up, self.down, self.left, self.right = values

    def __str__(self):
        return f"Policy(up={self.up}, down={self.down}, left={self.left}, right={self.right})"

    @staticmethod
    def kl_divergence(policy_at_state_p: 'SingleStatePolicy', policy_at_state_q: 'SingleStatePolicy') -> float:
        sum_v = 0.0
        for action_p, action_q in zip(policy_at_state_p.to_list(), policy_at_state_q.to_list()):
            sum_v += action_p * math.log2(action_p/action_q)
        return sum_v


class GridWorldPolicy:
    """
    GridWorldPolicy class represents a policy for a grid world environment.

    Attributes:
        - policy_grid (dict): A dictionary that maps states to individual state policies.
        - grid_world_env (GridWorldEnv): The grid world environment associated with the policy.

    Methods:
        - __init__(grid_world_env: GridWorldEnv, default_policy=(0.25, 0.25, 0.25, 0.25)):
            Initializes a new GridWorldPolicy instance.
        - get_policy(state: tuple) -> SingleStatePolicy:
            Returns the policy for a specific state.
        - action_probabilities(state: tuple) -> dict[Action, float]:
            Returns the probabilities of taking each action given a state.
        - interpolate(gamma):
            Interpolates policies based on gamma, creating a new PolicyGrid.
        - visualize():
            Visualizes the policy grid on the GridWorld layout.
        - _draw_arrows(ax, state, policy):
            Helper method to draw policy arrows on the grid.

    """
    def __init__(self, grid_world_env: GridWorldEnv, default_policy=(0.25, 0.25, 0.25, 0.25)) -> None:
        self.policy_grid = {}
        self.grid_world_env = grid_world_env

        # Initialize policy grid with default policy for each state
        for x in range(self.grid_world_env.grid_size_x):
            for y in range(self.grid_world_env.grid_size_y):
                self.policy_grid[(x, y)] = SingleStatePolicy(*default_policy)

    def get_policy(self, state: tuple):
        """Get the policy for a specific state."""
        return self.policy_grid[state]

    def action_probabilities(self, state: tuple) -> dict[Action, float]:
        """Return the probabilities of taking each action given a state."""
        policy_at_state = self.get_policy(state)
        return {
            Action.UP: policy_at_state.up,
            Action.DOWN: policy_at_state.down,
            Action.LEFT: policy_at_state.left,
            Action.RIGHT: policy_at_state.right
        }

    def interpolate(self, gamma):
        """Interpolate policies based on gamma, creating a new PolicyGrid."""
        new_policy_grid = GridWorldPolicy(self.grid_world_env)
        num_actions = len(Action)  # Number of possible actions

        for state, policy in self.policy_grid.items():
            original_policy_list = policy.to_list()
            interpolated_policy_list = [gamma * p + (1 - gamma) * (1 / num_actions) for p in original_policy_list]
            new_policy_grid.policy_grid[state].update_from_list(interpolated_policy_list)

        return new_policy_grid

    def visualize(self):
        """Visualize the policy grid on the GridWorld layout."""
        fig, ax = plt.subplots()

        # Define colors for each cell type
        cell_colors = {
            CellType.EMPTY: (192 / 255, 192 / 255, 192 / 255),  # Grey
            CellType.TARGET: (255 / 255, 215 / 255, 0 / 255),  # Yellow
            CellType.START: (0 / 255, 0 / 255, 255 / 255),  # Blue
            CellType.OBSTACLE: (0 / 255, 0 / 255, 0 / 255),  # Black
            CellType.TREASURE: (0 / 255, 255 / 255, 0 / 255),  # Green
            CellType.TRAP: (255 / 255, 0 / 255, 0 / 255)  # Red
        }

        # Define a mapping from cell type to integer
        cell_type_to_int = {
            CellType.EMPTY: 0,
            CellType.TARGET: 1,
            CellType.START: 2,
            CellType.OBSTACLE: 3,
            CellType.TREASURE: 4,
            CellType.TRAP: 5
        }

        # Create the colormap using the specified colors
        cmap = mcolors.ListedColormap([cell_colors[key] for key in cell_type_to_int.keys()])

        # Convert layout to numerical values for coloring
        data = np.zeros(self.grid_world_env.layout.shape, dtype=int)
        for i in range(self.grid_world_env.grid_size_x):
            for j in range(self.grid_world_env.grid_size_y):
                cell_type = self.grid_world_env.layout[i, j]
                data[i, j] = cell_type_to_int[cell_type]

        ax.imshow(data, cmap=cmap)

        # Display policy as arrows on the grid
        for state, policy in self.policy_grid.items():
            self._draw_arrows(ax, state, policy)

        plt.show()

    def _draw_arrows(self, ax, state, policy):
        """Helper method to draw policy arrows on the grid, skipping obstacles and targets."""
        i, j = state
        cell_type = self.grid_world_env.layout[i, j]

        # Skip drawing arrows for obstacles and targets
        if cell_type in [CellType.OBSTACLE, CellType.TARGET]:
            return

        arrow_scale = 0.3  # Scale factor for the arrow size
        head_width = 0.1  # Width of the arrow head
        head_length = 0.1  # Length of the arrow head
        arrow_color = 'white'  # Color of the arrow

        # Draw arrows based on the policy probabilities
        if policy.up > 0:
            ax.arrow(j, i, 0, -arrow_scale * policy.up, head_width=head_width, head_length=head_length, fc=arrow_color,
                     ec=arrow_color)
        if policy.down > 0:
            ax.arrow(j, i, 0, arrow_scale * policy.down, head_width=head_width, head_length=head_length, fc=arrow_color,
                     ec=arrow_color)
        if policy.left > 0:
            ax.arrow(j, i, -arrow_scale * policy.left, 0, head_width=head_width, head_length=head_length,
                     fc=arrow_color, ec=arrow_color)
        if policy.right > 0:
            ax.arrow(j, i, arrow_scale * policy.right, 0, head_width=head_width, head_length=head_length,
                     fc=arrow_color, ec=arrow_color)


# Example usage
if __name__ == "__main__":
    env = GridWorldEnv('layout.txt')
    policy_grid = GridWorldPolicy(env)
    # Example of setting a specific policy for a state
    policy_grid.policy_grid[(1, 1)].update_from_list([0.5, 0.1, 0.2, 0.2])
    # Visualize or use the policy grid as needed
    policy_grid.visualize()
