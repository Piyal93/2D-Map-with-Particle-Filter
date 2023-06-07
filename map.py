import random
import numpy as np
import matplotlib.pyplot as plt

current_true_position = (1,1)
measurement_noise_std = 0.2
num_particles = 100
frequency = 0.2

particles = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(num_particles)]

weights = [1.0 / num_particles] * num_particles

def measurement_model(position, noise_std):
    noise = random.gauss(0, noise_std)
    return [position[0] + noise, position[1] + noise]

# Define the motion model with sinusoidal trajectory
def motion_model(position, noise_std, i):
    x = position[0] + i
    y = position[1] + 2 * np.sin(0.3 * np.pi * frequency * x)
    noise = random.gauss(0, noise_std)
    return [x + noise, y + noise]

def simulated_motion_model(position, noise_std, i):
    x = i
    y = 0 + 2 * np.sin(np.pi * frequency * x)
    noise = random.gauss(0, noise_std)
    return (x, y + noise)


# Update particle weights based on the measurement: this func is responsible for updating the particle weights. It calculates the weight for each particle between the measurement and the particle position. The weights are then normalized to ensure they sum up to 1.
def update_weights(particles, measurement, noise_std):
    global weights 
    for i in range(num_particles):
        particle = particles[i]
        weight = 1.0 / (abs(measurement[0] - particle[0]) + abs(measurement[1] - particle[1]) + 1e-10)
        noise = random.gauss(0, noise_std)
        particles[i] = simulated_motion_model(particle, noise, i)
        weights[i] = weight

    # Normalizing the weights
    sum_weights = sum(weights)
    weights = [w / sum_weights for w in weights]

# Resample particles based on their weights:  this func performs resampling of particles based on their weights. It selects particles with higher weights more frequently and generates a new set of particles.
def resample(particles, weights):
    new_particles = []
    index = int(random.uniform(0, num_particles))
    beta = 0.0
    max_weight = max(weights)
    for _ in range(num_particles):
        beta += random.uniform(0, 2 * max_weight)
        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1) % num_particles
        new_particles.append(particles[index])
    return new_particles

# Simulate the particle filter process

motion_noise_std = 0.2
num_iterations = 15

# Define the map
fig, ax = plt.subplots()
ax.set_xlim([-1.5, 15.5])
ax.set_ylim([-10, 10])

# Plot the true position
# ax.plot(true_position[0], true_position[1], 'bx', label='True Position')
true_position_list = []
estimated_position_list = []

for i in range(num_iterations):
    # Generate a new simulated motion
    current_true_position = simulated_motion_model(current_true_position, measurement_noise_std, i)

    measurement = measurement_model(current_true_position, measurement_noise_std)

    # Update particle weights based on the measurement
    update_weights(particles, measurement, measurement_noise_std)

    # Resample particles
    particles = resample(particles, weights)

    # Estimate the current position by computing the mean of particles
    estimated_position = (sum([p[0] for p in particles]) / num_particles,
                          sum([p[1] for p in particles]) / num_particles)
    
    # estimated_position_list.append(estimated_position)
    true_position_list.append(current_true_position)
    # Plot the particles
    ax.scatter([p[0] for p in particles], [p[1] for p in particles], color='gray', s=1)

    # # Plot the estimated position
    #ax.plot(true_position[0], estimated_position[1], 'ro', label='Estimated Position')

    #x, y = zip(*estimated_position_list)
    #ax.plot(x, y, 'b--', alpha=1)

    xtrue , ytrue =  zip(*true_position_list)
    ax.plot(xtrue, ytrue, 'yellow', alpha=1)
    print(xtrue, ytrue)
    plt.pause(0.2)

    # Clear the plot for the next iteration
    #ax.clear()

    # Plot the map
    #ax.plot(map_x, map_y, 'k')

plt.show()

