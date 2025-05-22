import numpy as np
import matplotlib.pyplot as plt

def myPlot(DATA, net, num=None):
    w = net['weight']
    edge = net['edge']
    N, D = w.shape
    label = net['LabelCluster']

    plt.figure(num)
    plt.cla()

    # Plot data points
    plt.scatter(DATA[:, 0], DATA[:, 1], c='gray', s=3, alpha=0.5)

    # Plot edges
    for i in range(N - 1):
        for j in range(i + 1, N):
            if edge[i, j] > 0:
                plt.plot([w[i, 0], w[j, 0]], [w[i, 1], w[j, 1]], 'k', linewidth=1.5)

    # Plot nodes
    color = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0.85, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
        [0.635, 0.078, 0.184]
    ])
    m = color.shape[0]

    for k in range(N):
        plt.scatter(w[k, 0], w[k, 1], c=color[label[k] % m], s=350, marker='.')

    # Plot CountNode
    for i in range(N):
        countStr = str(net['CountNode'][i])
        plt.text(w[i, 0] + 0.01, w[i, 1] + 0.01, countStr, color='k', fontsize=8)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.grid(True)
    plt.axis('equal')
    plt.show()