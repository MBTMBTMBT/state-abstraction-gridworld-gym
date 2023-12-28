import numpy as np
from gridworld_env import GridWorldEnv
from gridworld_policy import GridWorldPolicy


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

