import numpy as np
import matplotlib.pyplot as plt

# Create the 2D space
fig, ax = plt.subplots()
ax.set_xlim([-1.5, 50])
ax.set_ylim([-10, 10])

# Trajectory List
trajectory_x = []
trajectory_y = []

# Set the initial position of the point
x, y = 0, 0

amplitude = 4
frequency = 0.3
phase = 0

#update function for the point in the trajectory
def update_state(x, y):
    x = x
    y = amplitude * np.sin(0.7 * np.pi * frequency * x + phase)
    return x, y


# Add Gaussian noise to the measurement
def noise(x, y):
    noise_x = np.random.normal(0, 5)
    noise_y = np.random.normal(0, 5)
    return x + noise_x, y + noise_y

# Define the particle filter function
def particle_filter(z, particles, weights):
    particles_new = np.zeros_like(particles)
    weights_new = np.zeros_like(weights)
    for i in range(len(particles)):
        x, y = particles[i]
        #t = np.random.uniform(0, 1)
        x, y = update_state(x, y)
        z_hat = noise(x, y)
        weights_new[i] = np.exp(-0.5 * ((z[0]-z_hat[0])**2/0.5**2 + (z[1]-z_hat[1])**2/0.5**2))
        particles_new[i] = x, y
    weights_new /= np.sum(weights_new)
    return particles_new, weights_new




# Create the point
point, = ax.plot(0, 0, 'o')



# Create the particles and weights for the particle filter
num_particles = 1000
particles = np.random.uniform(-100, 1000, (num_particles, 2))
weights = np.ones(num_particles) / num_particles



# Loop to update the position of the point and the particle filter every second
for i in range(50):
    # Update the position of the point
    x, y = update_state(i, y)
    z = noise(x, y)
    point.set_data(x, y)

    # Update the particle filter
    particles, weights = particle_filter(z, particles, weights)
    x_est, y_est = np.average(particles, axis=0, weights=weights)

    # Plot the estimated position of the point
    ax.plot(x_est, y_est, 'rx')

    # Add the new position to the trajectory list
    trajectory_x.append(x)
    trajectory_y.append(y)

    # Plot the trajectory
    ax.plot(trajectory_x, trajectory_y, color='orange', alpha=0.5)
    ax.scatter(x, y, color='blue')

    plt.draw()
    plt.pause(0.1)

plt.show()
