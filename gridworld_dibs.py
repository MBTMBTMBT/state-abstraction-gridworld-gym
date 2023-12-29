import numpy as np
import tqdm
import math

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
            for x in range(self.grid_size_x):
                for y in range(self.grid_size_y):
                    j = np.zeros(shape=(self.grid_size_x, self.grid_size_y))
                    for x1 in range(self.grid_size_x):
                        for y1 in range(self.grid_size_y):
                            state1 = (x1, y1)
                            j[state1] = math.log2(abs_stationary_distributions[state1])
                            state_policy_demo = self.demo_policy.get_policy(state1)
                            state_policy_abs = abs_policy.get_policy(state1)
                            j[state1] -= beta * SingleStatePolicy.kl_divergence(state_policy_demo, state_policy_abs)
                    max_position = np.unravel_index(np.argmax(j), j.shape)
