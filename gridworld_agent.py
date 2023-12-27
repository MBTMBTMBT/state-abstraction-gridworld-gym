import numpy as np
import gym
from gym import spaces
import time
from gridworld_env import *
from gridworld_policy import *
from gridworld_mdp import *

class GridWorldAgent:
    def __init__(self, env, policy_grid):
        self.env = env
        self.policy_grid = policy_grid

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


if __name__ == "__main__":
    # Create a GridWorldEnv environment
    env = GridWorldEnv('layout.txt')
    mdp = GridWorldMDP(env)
    mdp.value_iteration()
    policy_grid = mdp.derive_policy()

    # Create an agent with the environment and policy grid
    agent = GridWorldAgent(env, policy_grid)

    # Simulate the agent's behavior for a specified number of steps
    total_reward = agent.simulate(num_steps=100, render=True, refresh=0.5)
    print(f"Total reward from simulation: {total_reward}")
