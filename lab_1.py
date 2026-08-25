###
# Group Members
# Okwukwechukwu Mbajiorgu: 2430639
# Royal Goronga: 3014090
# Lulama Unity Hlungwani:3127972
# Adivhaho Nevondo:2580619
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    states = env.observation_space.n
    V = np.zeros(states)

    while True:
        delta = 0
        for s in range(states):
            v = V[s]
            new_v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:

                    next_val = 0 if done else V[next_state]  #Terminal state
                    new_v += action_prob * prob * (reward + discount_factor * next_val)

            V[s] = new_v

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            for probability, next_state, reward, done in env.P[state][action]:
                next_val = 0 if done else V[next_state]
                action_values[action] += probability * (
                    reward + discount_factor * next_val
                )

        return action_values

    policy = np.ones(
        [env.observation_space.n, env.action_space.n]
    ) / env.action_space.n

    while True:

        V = policy_evaluation_fn(env,policy,discount_factor)

        policy_stable = True
        new_policy = np.zeros_like(policy)

        for state in range(env.observation_space.n):

            action_values = one_step_lookahead(state, V)  # Store the expected value of each action

            best_action = np.argmax(action_values)

            new_policy[state, best_action] = 1.0

            old_action = np.argmax(policy[state])
            # If the best action differs from the previous action,
            # the policy has not yet converged
            if old_action != best_action:
                policy_stable = False

        policy = new_policy

        if policy_stable:
            return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        action_values = np.zeros(env.action_space.n)
        
        for action in range(env.action_space.n):
            for probability, next_state, reward, done in env.P[state][action]:
                next_val = 0 if done else V[next_state]
                action_values[action] += probability * (
                    reward + discount_factor * next_val
                )

        return action_values
    # For every state, calculate the value of all four actions, 
    # take the best one, update the state's value, 
    # and keep repeating until the values barely change.
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for state in range(env.observation_space.n):

            action_values = one_step_lookahead(state, V)  # Store the expected value of each action

            best_action_value = np.max(action_values)

            delta = max(delta, abs(best_action_value - V[state]))

            V[state] = best_action_value

        if delta < theta:
            break

    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for state in range(env.observation_space.n):
        policy[state, np.argmax(one_step_lookahead(state, V))] = 1.0

    return policy, V

def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # TODO: generate random policy

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    v = []

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # TODO: use  policy improvement to compute optimal policy and state values
    policy, v = [], []  # call policy_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # TODO: use  value iteration to compute optimal policy and state values
    policy, v = [], []  # call value_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


if __name__ == "__main__":
    main()
