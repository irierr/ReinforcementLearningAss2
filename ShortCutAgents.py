import random
import numpy as np

def choose_greedy(X,e):
    p = np.random.uniform()
    if p <= 1 - e:
        choices = np.argwhere(X == np.max(X)).flatten()
        a = np.random.choice(choices)  # Replace this with correct action selection
    else:
        choices = np.argwhere(X != np.max(X)).flatten()
        if choices.size > 0:
            a = np.random.choice(choices)
        else:
            a = np.random.randint(X.size)
    return a 

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        # TO DO: Add own code
        self.Q = np.zeros(shape=(self.n_states, self.n_actions))
        pass

    def select_action(self, state):
        # TO DO: Add own code
        # p = np.random.uniform()
        # if p <= 1 - self.epsilon:
        #     choices = np.argwhere(self.Q[state] == np.max(self.Q[state])).flatten()
        #     a = np.random.choice(choices)  # Replace this with correct action selection
        # else:
        #     choices = np.argwhere(self.Q[state] != np.max(self.Q[state])).flatten()
        #     if choices.size > 0:
        #         a = np.random.choice(choices)
        #     else:
        #         a = np.random.randint(self.n_actions)
        return choose_greedy(self.Q[state],self.epsilon)

    # TODO: check how to get next_state    
    def update(self, state, next_state, action, reward):
        # TO DO: Add own code
        self.Q[state, action] += self.alpha * (reward + np.max(self.Q[next_state]) - self.Q[state, action])


class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        # TO DO: Add own code
        self.Q = np.zeros(shape=(self.n_states, self.n_actions))


    def select_action(self, state):
        # TO DO: Add own code
        return choose_greedy(self.Q[state],self.epsilon)


    def update(self, state, next_state, action, reward):
        next_action = choose_greedy(self.Q[next_state],self.epsilon)
        self.Q[state,action] += self.alpha*(reward + self.Q[next_state, next_action] - self.Q[state,action])


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code
        pass

    def select_action(self, state):
        # TO DO: Add own code
        a = random.choice(range(self.n_actions))  # Replace this with correct action selection
        return a

    def update(self, state, action, reward):
        # TO DO: Add own code
        pass
