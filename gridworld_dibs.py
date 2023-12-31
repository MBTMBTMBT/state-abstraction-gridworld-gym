import numpy as np
import tqdm
import math
import matplotlib.pyplot as plt

from gridworld_agent import GridWorldAgent
from gridworld_env import GridWorldEnv
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
        abs_states[:, :, 0] = np.random.randint(self.grid_size_x, size=(self.grid_size_x, self.grid_size_y))
        abs_states[:, :, 1] = np.random.randint(self.grid_size_y, size=(self.grid_size_x, self.grid_size_y))

        # init abs policy:
        abs_policy = GridWorldPolicy(self.env)  # Actions are uniformly distributed by default.

        # init distribution:
        abs_stationary_distributions = np.ones(shape=(self.grid_size_x, self.grid_size_y))
        abs_stationary_distributions /= self.grid_size_x * self.grid_size_y  # rho_phi(s_phi) ~ Unif(1, |S|)

        for _ in tqdm.tqdm(range(max_iterations)):
            abs_states_copy = np.copy(abs_states)
            abs_stationary_distributions_copy = np.copy(abs_stationary_distributions)
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    j = np.zeros(shape=(self.grid_size_x, self.grid_size_y))
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            state1 = (x1, y1)
                            j[state1] = math.log2(abs_stationary_distributions[state1] + 1e-100)
                            state_policy_demo = self.demo_policy.get_policy(state1)
                            state_policy_abs = abs_policy.get_policy(state1)
                            j[state1] -= beta * SingleStatePolicy.kl_divergence(state_policy_demo, state_policy_abs)
                    abs_states_copy[x, y] = np.unravel_index(np.argmax(j), j.shape)
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    prob_sum = 0.0
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            if np.array_equal(abs_states_copy[x1, y1], np.array([x, y])):
                                prob_sum += abs_stationary_distributions[x1, y1]
                    abs_stationary_distributions_copy[x, y] = prob_sum
            max_delta = np.max(np.abs(abs_stationary_distributions - abs_stationary_distributions_copy))
            abs_states = abs_states_copy
            abs_stationary_distributions = abs_stationary_distributions_copy
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    state = (x, y)
                    state_policy_abs = abs_policy.get_policy(state)
                    numerator, denominator = np.zeros_like(np.array(state_policy_abs.to_list())), 1e-100
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            if np.array_equal(abs_states_copy[x1, y1], np.array([x, y])):
                                numerator += np.array(state_policy_abs.to_list()) * abs_stationary_distributions[x1, y1]
                                denominator += abs_stationary_distributions[x1, y1]
                    abs_policy.policy_grid[state].update_from_list(list(numerator / denominator))
            if max_delta < threshold:
                break

        return abs_states, abs_policy

    @staticmethod
    def visualize(abs_states, abs_policy):
        """Visualizes the abstract states and policy on the GridWorld layout."""

        # plot setup
        fig, ax = plt.subplots()

        # Create colormap for the abstract states
        unique_pairs = np.unique(abs_states.reshape(-1, 2), axis=0)
        num_unique_pairs = len(unique_pairs)
        cmap = plt.cm.get_cmap('tab20', num_unique_pairs)  # Choose a colormap with sufficiently distinct colors

        # Compute colors for abstract states
        indices = []
        for i in range(abs_states.shape[0]):
            for j in range(abs_states.shape[1]):
                index = np.where((unique_pairs == abs_states[i, j]).all(axis=1))
                indices.append(index[0][0])
        indices = np.array(indices)
        colors = indices.reshape(abs_states.shape[0], abs_states.shape[1])

        # Draw abstract states
        ax.imshow(colors, cmap=cmap)

        # Draw policy arrows
        for state, policy in abs_policy.policy_grid.items():
            # Get the action probabilities from the policy
            action_probabilities = policy.to_list()

            # compute lengths of arrows based on action probabilities
            arrow_lengths = [prob * 0.3 for prob in action_probabilities]  # Scale arrows by action probabilities

            # Draw the arrows
            i, j = state
            ax.arrow(j, i, 0, -arrow_lengths[0], color='white')
            ax.arrow(j, i, 0, arrow_lengths[1], color='white')
            ax.arrow(j, i, -arrow_lengths[2], 0, color='white')
            ax.arrow(j, i, arrow_lengths[3], 0, color='white')

        plt.show()


if __name__ == '__main__':
    from gridworld_mdp import GridWorldMDP
    # Create a GridWorldEnv environment
    env = GridWorldEnv('layout.txt')
    mdp = GridWorldMDP(env)
    mdp.value_iteration()
    policy_grid = mdp.derive_policy()

    dibs = GridWorldDibs(env, policy_grid)
    abs_states, abs_policy = dibs.run_dibs(beta=0.5, threshold=1e-10, max_iterations=100)
    # print(abs_states)
    dibs.visualize(abs_states, abs_policy)
