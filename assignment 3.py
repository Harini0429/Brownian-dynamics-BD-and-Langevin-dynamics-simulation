import numpy as np
import matplotlib.pyplot as plt

def non_dimensional_brownian_motion(T_star=100, dt_star=0.001):
    N = int(T_star / dt_star)
    x_star = np.zeros(N)
    y_star = np.zeros(N)
    z_star = np.zeros(N)
    n_x = np.random.uniform(-1, 1, N)
    n_y = np.random.uniform(-1, 1, N)
    n_z = np.random.uniform(-1, 1, N)
    for i in range(1, N):
        x_star[i] = x_star[i-1] + np.sqrt(6 * dt_star) * n_x[i]
        y_star[i] = y_star[i-1] + np.sqrt(6 * dt_star) * n_y[i]
        z_star[i] = z_star[i-1] + np.sqrt(6 * dt_star) * n_z[i]
    return x_star, y_star, z_star

def calculate_msd(x_star, y_star, z_star, max_delt):
    N = len(x_star)
    msd = np.zeros(max_delt + 1)
    for delt in range(max_delt + 1):
        nterms = 0
        for i in range(N - delt):
            msd[delt] += (x_star[i + delt] - x_star[i])**2 + (y_star[i + delt] - y_star[i])**2 + (z_star[i + delt] - z_star[i])**2
            nterms += 1
        msd[delt] /= nterms
    return msd

# Parameters
T_star = 100
dt_star = 0.001
max_delt = 100

# Simulate Brownian motion
x_star, y_star, z_star = non_dimensional_brownian_motion(T_star, dt_star)

# Plot the trajectory (xy projection)
plt.figure(figsize=(10, 8))
plt.plot(x_star, y_star, label='Trajectory')
plt.xlabel('X*')
plt.ylabel('Y*')
plt.title('2D Non-Dimensional Brownian Motion')
plt.legend()
plt.show()


def calculate_average_final_position(T_star, dt_star, iterations=20):
    final_positions = []
    for _ in range(iterations):
        x_star, y_star, z_star = non_dimensional_brownian_motion(T_star, dt_star)
        final_position = np.array([x_star[-1], y_star[-1], z_star[-1]])
        final_positions.append(final_position)
    final_positions = np.array(final_positions)
    average_final_position = np.mean(final_positions, axis=0)
    return average_final_position

iterations = 20

# Calculate average final position over many iterations
average_final_position = calculate_average_final_position(T_star, dt_star, iterations)

print(f'Average displacement: {average_final_position}')

# Calculate and plot MSD
msd = calculate_msd(x_star, y_star, z_star, max_delt)
t_star_values = np.arange(max_delt + 1) * dt_star

plt.figure(figsize=(10, 8))
plt.plot(t_star_values, msd, label='MSD')
plt.xlabel('t*')
plt.ylabel('MSD')
plt.title('Mean Square Displacement vs t*')
plt.legend()
plt.show()

# calculate slope
slope = (msd[1] - msd[0])/(t_star_values[1]-t_star_values[0])
print('Slope:',slope)

# Slope = 6*Diffusivity
# From this we get diffusivity