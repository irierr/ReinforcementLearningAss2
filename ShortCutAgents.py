import numpy as np


class Agent(object):
    def __init__(self, n_actions, n_states, epsilon, alpha):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q = np.zeros(shape=(self.n_states, self.n_actions))

    def select_action(self, state):
        return self.choose_greedy(self.Q[state])

    def choose_greedy(self, state):
        p = np.random.uniform()
        if p <= 1 - self.epsilon:
            choices = np.argwhere(state == np.max(state)).flatten()
            a = np.random.choice(choices)  # Replace this with correct action selection
        else:
            choices = np.argwhere(state != np.max(state)).flatten()
            if choices.size > 0:
                a = np.random.choice(choices)
            else:
                a = np.random.randint(state.size)
        return a


class QLearningAgent(Agent):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        super().__init__(n_actions, n_states, epsilon, alpha)

    def update(self, state, next_state, action, reward):
        self.Q[state, action] += self.alpha * (reward + np.max(self.Q[next_state]) - self.Q[state, action])


class SARSAAgent(Agent):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        super().__init__(n_actions, n_states, epsilon, alpha)

    def update(self, state, next_state, action, reward):
        next_action = super().choose_greedy(self.Q[next_state])
        self.Q[state, action] += self.alpha*(reward + self.Q[next_state, next_action] - self.Q[state, action])


class ExpectedSARSAAgent(Agent):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        super().__init__(n_actions, n_states, epsilon, alpha)

    def update(self, state, action, reward):
        # TO DO: Add own code
        pass
