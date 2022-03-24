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

    def choose_greedy(self, Q):
        p = np.random.uniform()
        if p <= 1 - self.epsilon:
            choices = np.argwhere(Q == np.max(Q)).flatten()
            a = np.random.choice(choices)
        else:
            choices = np.argwhere(Q != np.max(Q)).flatten()
            if choices.size > 0:
                a = np.random.choice(choices)
            else:
                a = np.random.randint(Q.size)
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
        self.Q[state, action] += self.alpha * (reward + self.Q[next_state, next_action] - self.Q[state, action])


class ExpectedSARSAAgent(Agent):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        super().__init__(n_actions, n_states, epsilon, alpha)

    def expected_epsilon_greedy(self, Q):
        probs_policy = np.zeros(self.n_actions)
        args_max = np.argwhere(Q == np.max(Q)).flatten()
        n_max = len(args_max)
        if n_max < self.n_actions:
            trues_max = np.full(self.n_actions, False, dtype=bool)
            trues_max[args_max] = True
            probs_policy[trues_max] = (1 - self.epsilon) / n_max
            probs_policy[~trues_max] = self.epsilon / (self.n_actions - n_max)
        else:
            probs_policy[:] = 1 / self.n_actions
        return np.sum(Q * probs_policy)

    def update(self, state, next_state, action, reward):
        self.Q[state, action] += self.alpha * (
                    reward + self.expected_epsilon_greedy(self.Q[next_state]) - self.Q[state, action])
