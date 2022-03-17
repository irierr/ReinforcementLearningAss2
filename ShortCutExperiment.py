# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
#%%
from ShortCutEnvironment import *
from ShortCutAgents import *

def main():
    SCE = ShortcutEnvironment() 
    n_states = SCE.state_size()
    n_actions = len(SCE.possible_actions())
    epsilon = 0.1
    alpha = 0.1
    QLA = QLearningAgent(n_actions, n_states, epsilon, alpha)
    for i in range(1000):
        current_state = SCE.state()
        while not SCE.done():
            a = QLA.select_action(current_state)
            r = SCE.step(a)
            next_state = SCE.state()
            QLA.update(current_state, next_state, a, r)
            current_state = next_state
        SCE.reset()    
    print(np.argmax(QLA.Q,axis=1).reshape((12,12)))


    
        
if __name__ == '__main__':
    main()

# %%
