import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ShortCutAgents import *
from ShortCutEnvironment import *


def plot_path(q_values, start_point, end_point, file_name):
    path = np.zeros(shape=q_values.shape)
    curr_point = start_point.copy()
    path[curr_point[0], curr_point[1]] = 2  # Set start point of path
    while curr_point != end_point:  # Follows path until end point reached
        if q_values[curr_point[0], curr_point[1]] == 0:  # Up
            curr_point[0] -= 1
        elif q_values[curr_point[0], curr_point[1]] == 1:  # Down
            curr_point[0] += 1
        elif q_values[curr_point[0], curr_point[1]] == 2:  # Left
            curr_point[1] -= 1
        elif q_values[curr_point[0], curr_point[1]] == 3:  # Right
            curr_point[1] += 1
        path[curr_point[0], curr_point[1]] = 1  # Update path
    path[curr_point[0], curr_point[1]] = 3  # Set end point of path

    fig, ax = plt.subplots()
    # Show grid
    plt.hlines(y=np.arange(0, 12)+0.5, xmin=np.full(12, 0)-0.5, xmax=np.full(12, 12)-0.5, color='black')
    plt.vlines(x=np.arange(0, 12)+0.5, ymin=np.full(12, 0)-0.5, ymax=np.full(12, 12)-0.5, color='black')

    col_map = ListedColormap(['white', 'yellow', 'blue', 'green'])  # Set colors
    ax.matshow(path, cmap=col_map)

    # Remove axis
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.savefig(f"plots/{file_name}.png")


def plot_reward_graph(rewards, alpha_values, file_name):
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    for i, r in enumerate(rewards):
        ax.plot(r, label=alpha_values[i])
    ax.legend(title="α values")
    fig.savefig(f"plots/{file_name}.png", dpi=300)


def run_episodes(n_episodes, environment, agent):
    rewards = np.zeros(shape=n_episodes)
    for i in range(n_episodes):
        current_state = environment.state()
        cum_reward = 0
        while not environment.done():
            a = agent.select_action(current_state)
            r = environment.step(a)
            cum_reward += r
            next_state = environment.state()
            agent.update(current_state, next_state, a, r)
            current_state = next_state
        environment.reset()
        rewards[i] = cum_reward
    return rewards


def run_repetitions(n_rep, n_episodes, epsilon, alpha, method, environment):
    rewards = np.zeros(shape=(n_rep, n_episodes))

    for i in range(n_rep):
        env = environment()
        n_states = env.state_size()
        n_actions = len(env.possible_actions())   
        agent = method(n_actions, n_states, epsilon, alpha)
        rewards[i] = run_episodes(n_episodes, env, agent)
    return np.mean(rewards, axis=0)


def run_repetitions_parallel(n_rep, n_episodes, epsilon, alpha, method, environment):
    results = []
    pool = Pool(processes=cpu_count())
    for i in range(n_rep):
        env = environment()
        n_states = env.state_size()
        n_actions = len(env.possible_actions())
        agent = method(n_actions, n_states, epsilon, alpha)
        results.append(pool.apply_async(run_episodes, args=(n_episodes, env, agent)))
    pool.close()
    pool.join()
    return np.mean([r.get() for r in results], axis=0)


def run_experiment(n_rep, n_episodes, epsilon, alpha_values, method, environment, parallel=True):
    results = np.zeros(shape=(len(alpha_values), n_episodes))
    for i, a in enumerate(alpha_values):
        if parallel:
            results[i] = run_repetitions_parallel(n_rep, n_episodes, epsilon, a, method, environment)
        else:
            results[i] = run_repetitions(n_rep, n_episodes, epsilon, a, method, environment)
        print(f"alpha value {a} done!")
    return results


def main():
    epsilon = 0.1
    alphas = [0.5, 0.9]#[0.01, 0.1, 0.5, 0.9]
    n_episodes = 10
    n_reps = 100 
    methods = [QLearningAgent, SARSAAgent, ExpectedSARSAAgent]
    methods_names = ['Q-Learning', 'SARSA', 'Expected_SARSA']
    environments = [ShortcutEnvironment, WindyShortcutEnvironment]
    environment_names = ['Shortcut', 'Windy_Shortcut']
    Rewards = {}
    start = time.time()
    for m in range(len(methods)):
        for e in range(len(environments)):
            mn, en = methods_names[m],environment_names[e]
            print(mn,en)
            Rewards[mn,en] = run_experiment(n_reps, n_episodes, epsilon, alphas, methods[m], environments[e], parallel=True)
            plot_reward_graph(Rewards[mn,en], alphas, mn+'_'+en)
    end = time.time()
    print(f"Whole experiment done in: {end - start}")

    

    # q_values_2d = np.argmax(QLA_Q, axis=1).reshape((12, 12))
    # plot_path(q_values_2d, [9, 2], [8, 8], "q_path")


if __name__ == '__main__':
    main()
