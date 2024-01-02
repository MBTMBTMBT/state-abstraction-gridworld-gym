import numpy as np
import tqdm
import math
import matplotlib.pyplot as plt
import copy

from gridworld_agent import GridWorldAgent
from gridworld_env import GridWorldEnv, CellType
from gridworld_policy import GridWorldPolicy, SingleStatePolicy


# class GridAbsStates:
#     def __init__(self, env: GridWorldEnv) -> None:
#         self.env = env
#         self.grid_size_x, self.grid_size_y = env.grid_size_x, env.grid_size_y
#
#         # x and y for positions, 2 for it saves another coord:
#         self.states = np.zeros(shape=(self.grid_size_x, self.grid_size_y, 2))
#
#
# class GridAbsPolicy:
#     def __init__(self, env: GridWorldEnv) -> None:
#         self.env = env
#         self.policy = GridWorldPolicy(self.env)


class GridWorldDibs:
    def __init__(self, env: GridWorldEnv, demo_policy: GridWorldPolicy):
        self.env = env
        self.grid_size_x, self.grid_size_y = env.grid_size_x, env.grid_size_y
        self.demo_policy = demo_policy
        self.demo_agent = GridWorldAgent(env, demo_policy)
        self.demo_stationary_distributions = self.demo_agent.sample_stationary_state_distribution()

    def run_dibs(self, beta: float, threshold: float, max_iterations: int):
        # Init abs states
        # x and y for positions, 2 for it saves another coord:
        abs_states = np.zeros(shape=(self.grid_size_x, self.grid_size_y, 2))
        # Fill abs_states with random states within grid size
        # abs_states[:, :, 0] = np.random.randint(self.grid_size_x, size=(self.grid_size_x, self.grid_size_y))
        # abs_states[:, :, 1] = np.random.randint(self.grid_size_y, size=(self.grid_size_x, self.grid_size_y))
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                abs_states[x, y] = np.array((x, y))

        # init distribution:
        abs_stationary_distributions = np.ones(shape=(self.grid_size_x, self.grid_size_y))
        abs_stationary_distributions /= self.grid_size_x * self.grid_size_y  # rho_phi(s_phi) ~ Unif(1, |S|)

        # init abs policy:
        abs_policy = GridWorldPolicy(self.env)  # Actions are uniformly distributed by default.
        abs_policy.policy_grid = copy.deepcopy(self.demo_policy.policy_grid)

        for _ in tqdm.tqdm(range(max_iterations)):
            abs_states_copy = np.copy(abs_states)
            abs_stationary_distributions_copy = np.copy(abs_stationary_distributions)
            pass
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    state = (x, y)
                    if not self.env.is_valid_state(state):
                        continue
                    j = np.zeros(shape=(self.grid_size_x, self.grid_size_y))
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            state1 = (x1, y1)
                            if not self.env.is_valid_state(state1):
                                j[state1] = - np.inf
                                continue
                            j[state1] = math.log2(abs_stationary_distributions[state] + 1e-100)
                            state_policy_demo = self.demo_policy.get_policy(state1)
                            state_policy_abs = abs_policy.get_policy(state)
                            j[state1] -= beta * SingleStatePolicy.kl_divergence(state_policy_demo, state_policy_abs)
                            # j_temp = j[state1]
                            pass
                    abs_states_copy[x, y] = np.unravel_index(np.argmax(j), j.shape)
                    j_argmax = np.argmax(j)
                    j_max = np.max(j)
                    max_state = abs_states_copy[x, y]
                    pass
            pass
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    prob_sum = 0.0
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            if np.array_equal(abs_states[x, y], np.array(abs_states[x1, y1])):
                                prob_sum += abs_stationary_distributions[x1, y1]
                    abs_stationary_distributions_copy[x, y] = prob_sum
            max_delta = np.max(np.abs(abs_stationary_distributions - abs_stationary_distributions_copy))
            abs_states = abs_states_copy
            abs_stationary_distributions = abs_stationary_distributions_copy
            pass
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    state = (x, y)
                    state_policy_abs = abs_policy.get_policy(state)
                    numerator, denominator = np.zeros_like(np.array(state_policy_abs.to_list())), 1e-100
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            if np.array_equal(abs_states[x, y], np.array(abs_states[x1, y1])):
                                state1 = (x1, y1)
                                numerator += np.array(self.demo_policy.get_policy(state1).to_list()) * abs_stationary_distributions[x1, y1]
                                denominator += abs_stationary_distributions[x1, y1]
                    policy_ = list(numerator / denominator)
                    abs_policy.policy_grid[state].update_from_list(list(numerator / denominator))

            if max_delta < threshold:
                break

        return abs_states, abs_policy

    def visualize(self, abs_states, abs_policy):
        """Visualizes the abstract states and policy on the GridWorld layout."""

        # plot setup
        fig, ax = plt.subplots()

        # Create colormap for the abstract states
        unique_pairs = np.unique(abs_states.reshape(-1, 2), axis=0)
        num_unique_pairs = len(unique_pairs)
        cmap = plt.cm.get_cmap('tab20', num_unique_pairs)  # Original colormap for valid cells

        # Initialize an array to store RGB color for each cell
        colors = np.zeros((abs_states.shape[0], abs_states.shape[1], 3))

        # RGB color for black
        black = np.array([0, 0, 0])

        for i in range(abs_states.shape[0]):
            for j in range(abs_states.shape[1]):
                if not self.env.is_valid_state((i, j)):  # Check if the cell is invalid
                    colors[i, j] = black  # Set gray color for invalid cells
                else:
                    index = np.where((unique_pairs == abs_states[i, j]).all(axis=1))[0][0]
                    colors[i, j] = cmap(index)[:3]  # Use colormap for valid cells

        # Draw abstract states
        ax.imshow(colors)

        # Display policy as arrows on the grid
        for state, policy in abs_policy.policy_grid.items():
            self._draw_arrows(abs_policy, ax, state, policy)

        plt.show()

    @staticmethod
    def _draw_arrows(abs_policy, ax, state, policy):
        """Helper method to draw policy arrows on the grid, skipping obstacles and targets."""
        i, j = state
        cell_type = abs_policy.grid_world_env.layout[i, j]

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


if __name__ == '__main__':
    from gridworld_mdp import GridWorldMDP

    # Create a GridWorldEnv environment
    env = GridWorldEnv('layout.txt')
    mdp = GridWorldMDP(env)
    mdp.value_iteration()
    policy_grid = mdp.derive_policy().add_noise(0.25)
    # policy_grid.interpolate(gamma=0.8)

    dibs = GridWorldDibs(env, policy_grid)
    abs_states, abs_policy = dibs.run_dibs(beta=1e-10, threshold=0.25e-100, max_iterations=1000)
    # print(abs_states)
    dibs.visualize(abs_states, abs_policy)
