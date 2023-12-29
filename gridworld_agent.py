import numpy as np
import time
import tqdm
from gridworld_mdp import GridWorldPolicy
from gridworld_env import GridWorldEnv, Action
from gridworld_mdp import GridWorldMDP


class GridWorldAgent:
    """

    Class: GridWorldAgent

    The GridWorldAgent class represents an agent in the GridWorld environment. It is responsible for choosing actions based on a given policy and simulating the agent's behavior in the environment
    *.

    Attributes:
    - env: The GridWorldEnv object representing the environment the agent is in.
    - policy_grid: The GridWorldPolicy object representing the policy to be followed by the agent.

    Methods:
    - __init__(self, env: GridWorldEnv, grid_world_policy: GridWorldPolicy)
        Constructs a new GridWorldAgent object with the given environment and policy_grid.

    - choose_action(self, state)
        Chooses an action based on the policy probabilities for the given state.

    - simulate(self, num_steps=100, render=False, refresh=0)
        Simulates the agent's behavior in the environment for a specified number of steps.
        Returns the total reward accumulated during the simulation.

    - sample_stationary_state_distribution(self, num_steps=50, num_iterations=500)
        Samples the stationary state distribution of the agent's behavior in the environment.
        Returns a numpy array representing the stationary state distribution.

    Note: This class assumes the existence of other classes such as GridWorldEnv and GridWorldPolicy, which are not defined in this documentation.

    """
    def __init__(self, env: GridWorldEnv, grid_world_policy: GridWorldPolicy):
        self.env = env
        self.policy_grid = grid_world_policy

    def choose_action(self, state):
        # Retrieve the policy for the current state
        policy = self.policy_grid.get_policy(state)
        action_probabilities = policy.to_list()
        
        # Choose an action based on the policy probabilities
        action = np.random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT], p=action_probabilities)
        return action

    def simulate(self, num_steps=100, render=False, refresh=0):
        state = self.env.reset()
        total_reward = 0
        
        for _ in range(num_steps):
            start_time = time.time()
            action = self.choose_action(tuple(state))
            next_state, reward, done, _ = self.env.step(action.value)
            
            # Accumulate the reward
            total_reward += reward
            
            # Update the state
            state = next_state
            
            if render:
                self.env.render()
            
            # Calculate the elapsed time and wait if necessary
            elapsed_time = time.time() - start_time
            if elapsed_time < refresh:
                time.sleep(refresh - elapsed_time)

            # Exit if we reach a terminal state
            if done:
                break
        
        return total_reward

    def sample_stationary_state_distribution(self, num_steps=50, num_iterations=500):
        total_steps = 0
        stationary_state_distribution = np.zeros(shape=(self.env.grid_size_x, self.env.grid_size_y))
        for _ in tqdm.tqdm(range(num_iterations)):
            state = self.env.reset()
            for _ in range(num_steps):
                stationary_state_distribution[tuple(state)] += 1
                total_steps += 1
                action = self.choose_action(tuple(state))
                next_state, reward, done, _ = self.env.step(action.value)

                # Update the state
                state = next_state

                # Exit if we reach a terminal state
                if done:
                    stationary_state_distribution[tuple(state)] += 1
                    total_steps += 1
                    break
        stationary_state_distribution /= total_steps
        return stationary_state_distribution


if __name__ == "__main__":
    # Create a GridWorldEnv environment
    env = GridWorldEnv('layout.txt')
    mdp = GridWorldMDP(env)
    mdp.value_iteration()
    policy_grid = mdp.derive_policy()

    # Create an agent with the environment and policy grid
    agent = GridWorldAgent(env, policy_grid)

    # Simulate the agent's behavior for a specified number of steps
    # total_reward = agent.simulate(num_steps=100, render=True, refresh=0.5)
    # print(f"Total reward from simulation: {total_reward}")

    stationary_state_distribution = agent.sample_stationary_state_distribution(num_steps=50, num_iterations=int(1e5))
    print(stationary_state_distribution)
