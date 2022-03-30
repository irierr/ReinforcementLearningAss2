import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ShortCutAgents import *
from ShortCutEnvironment import *
from tqdm import tqdm


def plot_path(q_values, start_points, end_point, file_name, windy):
    fig, ax = plt.subplots()
    # Show grid
    plt.hlines(y=np.arange(0, 12) + 0.5, xmin=np.full(12, 0) - 0.5, xmax=np.full(12, 12) - 0.5, color='black')
    plt.vlines(x=np.arange(0, 12) + 0.5, ymin=np.full(12, 0) - 0.5, ymax=np.full(12, 12) - 0.5, color='black')
    path = np.zeros(shape=(12, 12))

    q_max = np.argmax(q_values, axis=1).reshape((12, 12))

    for start_point in start_points:
        curr_point = start_point.copy()
        path[curr_point[0], curr_point[1]] = 2  # Set start point of path
        if not windy:
            while curr_point != end_point:  # Follows path until end point reached
                if q_max[curr_point[0], curr_point[1]] == 0:  # Up
                    curr_point[0] -= 1
                elif q_max[curr_point[0], curr_point[1]] == 1:  # Down
                    curr_point[0] += 1
                elif q_max[curr_point[0], curr_point[1]] == 2:  # Left
                    curr_point[1] -= 1
                elif q_max[curr_point[0], curr_point[1]] == 3:  # Right
                    curr_point[1] += 1
                path[curr_point[0], curr_point[1]] = 1  # Update path
    path[end_point[0], end_point[1]] = 3  # Set end point of path

    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

    col_map = ListedColormap(['white', 'yellow', 'blue', 'green'])  # Set colors
    ax.matshow(path, cmap=col_map)

    # Add action labels
    for (i, j), z in np.ndenumerate(format_greedy_actions(q_values)):
        ax.text(j, i, z, ha='center', va='center')

    # Remove axis
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.savefig(f"plots/paths/{file_name}.png", dpi=300)


def format_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12, 12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions == 0] = '↑'
    print_string[greedy_actions == 1] = '↓'
    print_string[greedy_actions == 2] = '←'
    print_string[greedy_actions == 3] = '→'
    print_string[np.max(Q, 1).reshape((12, 12)) == 0] = ' '
    return print_string


def plot_reward_graph(rewards, alpha_values, file_name):
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    for i, r in enumerate(rewards):
        ax.plot(r, label=alpha_values[i])
    ax.legend(title="α values")
    fig.savefig(f"plots/rewards/{file_name}.png", dpi=300)


def run_episodes(n_episodes, environment, agent, disable_tqdm=True):
    rewards = np.zeros(shape=n_episodes)
    for i in tqdm(range(n_episodes), disable=disable_tqdm, desc=f"{agent.__class__.__name__} {environment.__class__.__name__}"):
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


def run_repetitions(n_rep, n_episodes, epsilon, alpha, agent_method, environment):
    rewards = np.zeros(shape=(n_rep, n_episodes))
    env = environment()
    n_states = env.state_size()
    n_actions = len(env.possible_actions())
    for i in range(n_rep):
        agent = agent_method(n_actions, n_states, epsilon, alpha)
        rewards[i] = run_episodes(n_episodes, env, agent)
        env.reset()
    return np.mean(rewards, axis=0)


def run_repetitions_parallel(n_rep, n_episodes, epsilon, alpha, agent_method, environment):
    async_results = []
    pool = Pool(processes=cpu_count())
    for i in range(n_rep):
        env = environment()
        agent = agent_method(len(env.possible_actions()), env.state_size(), epsilon, alpha)
        async_results.append(pool.apply_async(run_episodes, args=(n_episodes, env, agent)))
    pool.close()
    pool.join()
    return np.mean([r.get() for r in async_results], axis=0)


def run_experiment(n_rep, n_episodes, epsilon, alpha_values, agent, environment, parallel=True):
    means = np.zeros(shape=(len(alpha_values), n_episodes))
    for i, a in enumerate(tqdm(alpha_values, desc=f"{agent.__name__} {environment.__name__} for {alpha_values}")):
        if parallel:
            means[i] = run_repetitions_parallel(n_rep, n_episodes, epsilon, a, agent, environment)
        else:
            means[i] = run_repetitions(n_rep, n_episodes, epsilon, a, agent, environment)
    return means


def main():
    # rewards plots
    epsilon = 0.1
    alphas = [0.01, 0.1, 0.5, 0.9]
    n_episodes = 1000
    n_reps = 100
    agents = [QLearningAgent, SARSAAgent, ExpectedSARSAAgent]
    agents_names = ['Q-Learning', 'SARSA', 'Expected_SARSA']
    rewards = {}
    start = time.time()
    for a in range(len(agents)):
        an = agents_names[a]
        rewards[an] = run_experiment(n_reps, n_episodes, epsilon, alphas, agents[a], ShortcutEnvironment)
        plot_reward_graph(rewards[an], alphas, f"{an}_rewards")
        np.save(f"arrays/reward_arrays/{an}", rewards[an])

    # Q(s, a) plots
    alpha = 0.1
    n_episodes = 10000
    environments = [ShortcutEnvironment, WindyShortcutEnvironment]
    environment_names = ['Shortcut', 'Windy_Shortcut']
    windy = False
    for a in range(len(agents)):
        for e in range(len(environments)):
            an, en = agents_names[a], environment_names[e]
            if an == 'Expected_SARSA' and en == 'Windy_Shortcut':
                continue
            env = environments[e]()
            agent = agents[a](len(env.possible_actions()), env.state_size(), epsilon, alpha)
            _ = run_episodes(n_episodes, env, agent, disable_tqdm=False)
            if en == 'Windy_Shortcut':
                windy = True
            plot_path(agent.Q, [[9, 2], [2, 2]], [8, 8], f"{an}_{en}_q_path", windy=windy)
            windy = False
            np.save(f"arrays/Q_Arrays/{an}_{en}", agent.Q)
    end = time.time()
    print(f"Whole experiment done in: {end - start}")


if __name__ == '__main__':
    main()
