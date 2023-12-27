import numpy as np
from gridworld_env import *
from gridworld_policy import *


class GridWorldMDP:
    """Class to represent a Markov Decision Process for GridWorldEnv."""

    def __init__(self, env, gamma=0.99):
        """
        Initialize the GridWorld MDP.

        Parameters:
        env (GridWorldEnv): The GridWorld environment.
        gamma (float): The discount factor for future rewards.
        """
        self.env = env
        self.gamma = gamma
        self.value_function = np.zeros((env.grid_size_x, env.grid_size_y))

    def value_iteration(self, theta=1e-10):
        """
        Perform value iteration to compute the optimal value function.
        Parameters:
        theta (float): Threshold for determining convergence.
        """
        while True:
            delta = 0
            # Create a copy of the current value function
            V_copy = np.copy(self.value_function)
            for i in range(self.env.grid_size_x):
                for j in range(self.env.grid_size_y):
                    state = (i, j)
                    # Skip terminal states
                    if not self.env.is_valid_state(state):
                        continue
                    if self.env.get_type(state) == CellType.TARGET:
                        self.value_function[state] = self.env.get_reward(state)
                        continue
                    # Initialize max value as negative infinity
                    max_value = float('-inf')
                    # Loop through all actions to find the best action value
                    for a in Action:
                        expected_value = 0
                        transitions = self.env.get_transitions(state, a)
                        # Sum over all possible next states
                        for next_state, prob in transitions:
                            expected_value += prob * V_copy[next_state]
                        # Select the max value across all actions
                        max_value = max(max_value, expected_value)
                    # Update the value function for the state
                    self.value_function[i, j] = self.env.get_reward(state) + self.gamma * max_value
                    # Update the delta
                    delta = max(delta, abs(V_copy[i, j] - self.value_function[i, j]))
            # Check for convergence
            if delta < theta:
                break

    def derive_policy(self):
        """
        Derive the optimal policy from the value function.

        Returns:
        PolicyGrid: The policy grid representing the optimal policy.
        """
        policy_grid = PolicyGrid(self.env)
        for i in range(self.env.grid_size_x):
            for j in range(self.env.grid_size_y):
                state = (i, j)
                # Skip terminal states
                if state in self.env.get_terminal_states():
                    continue

                best_actions = []
                best_value = float('-inf')

                # Evaluate each action's value
                for action in Action:
                    action_value = 0
                    transitions = self.env.get_transitions(state, action)
                    for next_state, prob in transitions:
                        action_value += prob * self.value_function[next_state]
                    
                    # Keep track of the best actions
                    if action_value > best_value:
                        best_actions = [action]
                        best_value = action_value
                    elif action_value == best_value:
                        best_actions.append(action)

                # Update the policy for the current state
                current_policy = policy_grid.get_policy(state)
                num_best_actions = len(best_actions)
                for action in Action:
                    action_prob = 1.0 / num_best_actions if action in best_actions else 0.0
                    setattr(current_policy, action.name.lower(), action_prob)

        return policy_grid
    
    def visualize(self):
        """Visualize the grid with the values of each state."""
        fig, ax = plt.subplots()

        # Define colors for each cell type
        cell_colors = {
            CellType.EMPTY: (192/255, 192/255, 192/255),    # Grey
            CellType.TARGET: (255/255, 215/255, 0/255),     # Yellow
            CellType.START: (0/255, 0/255, 255/255),        # Blue
            CellType.OBSTACLE: (0/255, 0/255, 0/255),       # Black
            CellType.TREASURE: (0/255, 255/255, 0/255),     # Green
            CellType.TRAP: (255/255, 0/255, 0/255)          # Red
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
        data = np.zeros(self.env.layout.shape, dtype=int)
        for i in range(self.env.grid_size_x):
            for j in range(self.env.grid_size_y):
                cell_type = self.env.layout[i, j]
                data[i, j] = cell_type_to_int[cell_type]

        ax.imshow(data, cmap=cmap)

        # Display the value of each state
        for i in range(self.env.grid_size_x):
            for j in range(self.env.grid_size_y):
                value_text = f"{self.value_function[i, j]:.1f}"
                ax.text(j, i, value_text, ha='center', va='center', color='white')

        plt.show()


# Usage example
if __name__ == "__main__":
    env = GridWorldEnv('layout.txt')
    mdp = GridWorldMDP(env)

    mdp.value_iteration()
    mdp.visualize()

    policy_grid = mdp.derive_policy()
    policy_grid.visualize()
