import numpy as np
import matplotlib.pyplot as plt

def estimateDensityByCountNode(weight, count_node):
    node_positions = weight

    # Bandwidth calculation using Silverman's Rule
    n = node_positions.shape[0]
    sigma_x = np.std(node_positions[:, 0])
    sigma_y = np.std(node_positions[:, 1])

    h_x = sigma_x * (4 / (3 * n)) ** (1/5)
    h_y = sigma_y * (4 / (3 * n)) ** (1/5)

    # Set the grid range and resolution
    grid_size = 100  # Grid resolution
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    density = np.zeros_like(X)

    # Calculate density for each grid point
    for i in range(grid_size):
        for j in range(grid_size):
            grid_point = np.array([X[i, j], Y[i, j]])
            distances_x = (node_positions[:, 0] - grid_point[0]) / h_x
            distances_y = (node_positions[:, 1] - grid_point[1]) / h_y
            kernel_values = count_node * np.exp(-(distances_x**2 + distances_y**2) / 2)
            density[i, j] = np.sum(kernel_values)

    # Normalize density to the range [0, 1]
    density = density / np.max(density)

    # Plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, density, cmap='jet', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    ax.set_title('Estimated Density by CountNode')
    plt.show()