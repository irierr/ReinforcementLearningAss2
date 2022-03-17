from ShortCutEnvironment import *
from ShortCutAgents import *
from matplotlib import colors
import matplotlib.pyplot as plt


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
    cmap = colors.ListedColormap(['white', 'yellow', 'blue', 'green'])  # Set colors
    # Show grid
    plt.hlines(y=np.arange(0, 12)+0.5, xmin=np.full(12, 0)-0.5, xmax=np.full(12, 12)-0.5, color='black')
    plt.vlines(x=np.arange(0, 12)+0.5, ymin=np.full(12, 0)-0.5, ymax=np.full(12, 12)-0.5, color='black')
    # Remove axis
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.matshow(path, cmap=cmap)
    plt.savefig(file_name)


def main():
    SCE = ShortcutEnvironment()
    n_states = SCE.state_size()
    n_actions = len(SCE.possible_actions())
    epsilon = 0.1
    alpha = 0.1
    QLA = QLearningAgent(n_actions, n_states, epsilon, alpha)
    for i in range(10000):
        current_state = SCE.state()
        while not SCE.done():
            a = QLA.select_action(current_state)
            r = SCE.step(a)
            next_state = SCE.state()
            QLA.update(current_state, next_state, a, r)
            current_state = next_state
        SCE.reset()

    q_values_2d = np.argmax(QLA.Q, axis=1).reshape((12, 12))
    plot_path(q_values_2d, [9, 2], [8, 8], "path.png")


if __name__ == '__main__':
    main()