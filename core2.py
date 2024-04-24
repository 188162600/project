
   # Import everything from question 1
from core1 import *

def main():
         # Constants and initial conditions
    G = 6.67430e-11  # Gravitational constant
    dt = 60 * 60  # Time step (1 hour)
    period = 365.256 * 24 * 3600  # Orbital period of Earth (1 year in seconds)
    total_time = period  # Total simulation time
    times = np.arange(0, total_time, dt)

    # Masses of Sun and Earth, positions, and velocities
    m = np.array([1.989e30, 5.972e24])  # Masses in kg
    distance = 149.6e9  # Distance in metres (1 AU) of the average radius of orbit of Earth
    x = np.array([[0, 0], [distance, 0]])  # Initial positions

    circumference = np.pi * distance * 2  # Orbit circumference
    velocity = circumference / period  # Orbital velocity (m/s)
    print(velocity)
    # velocity=29800
    v = np.array([[0, 0], [0, velocity]])  # Initial velocities

    # Simulation loop to produce set of positions
    positions = np.zeros((len(times), 2, 2))  # To store positions for plotting
    for i in range(len(times)):
        positions[i] = x
        verlet_update_all(x, v, m, dt, G,[False,True]) # Produces trajectories

    # Plot the trajectories
    plt.figure(figsize=(10, 5))
    plt.plot(positions[:, 0, 0], positions[:, 0, 1], 'o-', label='The Sun')
    plt.plot(positions[:, 1, 0], positions[:, 1, 1], 'o-', label='The Earth')
    plt.title("Trajectories of the Sun and Earth")
    plt.xlabel("X Position (metres)")
    plt.ylabel("Y Position (metres)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()
    