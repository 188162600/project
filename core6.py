
    # Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Import everything from question 1
from core1 import *

def main():
    G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
    dt = 3600*1  # Time step (1 hour)
    total_time = 3600 * 24 * 365 /20
    times = np.arange(0, total_time, dt)

    # Masses of the stars
    m1 = 1.989e30  # Mass of the first star (similar to the Sun)
    m2 = 1.1 * m1  # Mass of the second star
    m3 = 0.9 * m1  # Mass of the third star
    m = np.array([m1, m2, m3])

    # Initial positions (x, y) in meters
    x = np.array([[0, 0], [1.5e10, 0], [0, 1.5e10]])  # Setting each star 1.5e10 away from eachother

    # Orbital velocities relative to the masses
    r1 = np.linalg.norm(x[1] - x[0])  # Distance between stars 1 and 2
    r2 = np.linalg.norm(x[2] - x[1])  # Distance between stars 2 and 3
    v1 = np.sqrt(G * m1 / r1)  # Orbital velocity for star 1
    v2 = np.sqrt(G * m2 / r2)  # Orbital velocity for star 2
    v3 = np.sqrt(G * m3 / r2)  # Orbital velocity for star 3
    v = np.array([[0, 0], [0, v1], [0, v2]])  # Initial velocities

    # Prepare to plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.6e11, 1.6e11)
    ax.set_ylim(-1.6e11, 1.6e11)
    ax.set_title("Orbital Simulation: Sun-like Stars")
    ax.set_xlabel("X Position (meters)")
    ax.set_ylabel("Y Position (meters)")
    ax.grid(True)

    # Plotting elements
    lines = [ax.plot([], [], 'o-', label=f'Star {i+1}')[0] for i in range(3)]

    # Initialize the animation
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # Animation update function
    def update(frame):
        # Assume verlet_update_all correctly updates x, v
        verlet_update_all(x, v, m, dt, G)
        for i, line in enumerate(lines):
            line.set_data(x[:, 0], x[:, 1])
        return lines

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=True)
    plt.legend()
    # plt.show()
    from IPython.display import HTML, display,clear_output  # Import display function
    print(f"rendering {len(times)} ")
    plt.rcParams['animation.embed_limit'] = 100.0
    clear_output(wait=True)  # Clears the output of the static image
    display(HTML(ani.to_jshtml()))

    # Optionally, you can close the figure to ensure it doesn't consume memory
    plt.close(fig)
if __name__ == "__main__":
    main()
    
    