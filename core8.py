# Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Import everything from question 1
from core1 import *


def escape_velocity(mass, radius,G):
 
    return np.sqrt(2 * G * mass / radius)
def energy(m,v):
    return 0.5*m*v**2
def gravitational_acceleration(m2, x1, x2, G=6.67430e-11):
    r_vec = x2 - x1  # displacement vector pointing from x1 to x2
    r = np.linalg.norm(r_vec)  # Euclidean distance between the two positions
    if r == 0:
        raise ValueError("Collision or zero distance between objects.")
    return G * m2 / r**3 * r_vec  # Gravitational acceleration vector pointing toward the other mass

# Define the Verlet step for gravity
def verlet_step_gravity(x, v, m_others, x_others, dt, G):
    a = np.sum([gravitational_acceleration(m_other, x, x_other, G) for x_other, m_other in zip(x_others, m_others)], axis=0)
    # print(a)
    x_new = x + v * dt + 0.5 * a * dt**2
    a_new = np.sum([gravitational_acceleration(m_other, x_new, x_other, G) for x_other, m_other in zip(x_others, m_others)], axis=0)
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new

# Define the update function for all objects
def verlet_update_all(x, v, m, dt, G, update=True):
    n_objects = x.shape[0]
  
    if isinstance(update, bool):
        update = [update] * n_objects
    if isinstance(dt, (int, float)):
        dt = [dt] * n_objects
    
    x_new = np.copy(x)
    v_new = np.copy(v)
    for i in range(n_objects):
        x_others = np.vstack([x[j] for j in range(n_objects) if j != i])
        m_others = np.array([m[j] for j in range(n_objects) if j != i])
        if update[i]:
            x_new[i], v_new[i] = verlet_step_gravity(x[i], v[i], m_others, x_others, dt[i], G)
    x[:], v[:] = x_new, v_new

# Define the gravitational constant
G = 6.67430e-11  # m^3 kg^-1 s^-2
AU = 1.496e11
planet_data = {
    'Sun': (1.989e30, np.array([0, 0]), np.array([0, 0]),  10,1), 
    'Mercury': (3.30e23, np.array([5.79e10, 0]), np.array([0, 47.87e3]),3,1),
    'Venus': (4.87e24, np.array([1.08e11, 0]), np.array([0, 35.02e3]), 4,1),
    'Earth': (5.97e24, np.array([1.50e11, 0]), np.array([0, 29.78e3]), 4,1),
    'Mars': (6.42e23, np.array([2.28e11, 0]), np.array([0, 24.07e3]), 3,1),
    'Jupiter': (1.90e27, np.array([7.78e11, 0]), np.array([0, 13.07e3]), 7,1),
    'Saturn': (5.68e26, np.array([1.43e12, 0]), np.array([0, 9.69e3]), 6,1),
    'Uranus': (8.68e25, np.array([2.87e12, 0]), np.array([0, 6.81e3]), 5,1),
    'Neptune': (1.02e26, np.array([4.50e12, 0]), np.array([0, 5.43e3]), 6,1),
    'Close-orbit Exoplanet': (1.90e27, np.array([0.05 * AU, 0]), np.array([0, np.sqrt(6.67430e-11 * 1.989e30 / (0.05 * AU))]), 2,1) 
    }
update_freq=np.zeros(len(planet_data))
for value,i in enumerate(planet_data.values()):
    update_freq[value]=i[4]


def main():


    dt = 3600 * 24
    dt_for_each_planet = dt*update_freq
    
    total_time = 3600 * 24 * 365 * 2
    times = np.arange(0, total_time, dt)

    masses = np.array([data[0] for data in planet_data.values()])
    positions = np.array([data[1] for data in planet_data.values()])
    velocities = np.array([data[2] for data in planet_data.values()])
    sizes = np.array([data[3] for data in planet_data.values()])  

    # Prepare to plot
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim(-6e12, 6e12)
    ax.set_ylim(-6e12, 6e12)

    # Plot elements: planets, orbits, and labels
    planets = [ax.plot([], [], 'o', markersize=sizes[i], label=name)[0] for i, name in enumerate(planet_data.keys())]
    orbits = [ax.plot([], [], '-', linewidth=0.5)[0] for _ in planet_data.keys()]
    labels = [ax.text(0, 0, name, fontsize=9) for name in planet_data.keys()]

    def init():
        for planet, orbit in zip(planets, orbits):
            planet.set_data([], [])
            orbit.set_data([], [])
        for label in labels:
            label.set_position((0, 0))
        return planets + orbits + labels

    def animate(i):
        nonlocal positions, velocities
        verlet_update_all(positions, velocities, masses, dt_for_each_planet, G,update=i%update_freq==0)
        for planet, orbit, label, pos, i in zip(planets, orbits, labels, positions,range(len(planets))):
#             if i%update_freq:
#                 continue
            
            x, y = pos
            planet.set_data(x, y)
            old_x, old_y = orbit.get_data()
            orbit.set_data(np.append(old_x, x), np.append(old_y, y))
            label.set_position((x, y))
        return planets + orbits + labels

    ani = FuncAnimation(fig, animate, frames=len(times), init_func=init, interval=50, blit=True)
#     plt.rcParams["animation.html"] = "jshtml"
    plt.rcParams['animation.embed_limit'] = 200.0
    ax.legend(loc='upper right')

    from IPython.display import HTML, display,clear_output  # Import display function
    print(f"rendering {len(times)} steps with update freq to each plane of {update_freq}")
    clear_output(wait=True)  # Clears the output of the static image
    display(HTML(ani.to_jshtml()))
    # Optionally, you can close the figure to ensure it doesn't consume memory
    plt.close(fig)
    
if __name__ == "__main__":
    main()
# Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Import everything from question 1
from core1 import *


def escape_velocity(mass, radius,G):
 
    return np.sqrt(2 * G * mass / radius)
def energy(m,v):
    return 0.5*m*v**2
def gravitational_acceleration(m2, x1, x2, G=6.67430e-11):
    r_vec = x2 - x1  # displacement vector pointing from x1 to x2
    r = np.linalg.norm(r_vec)  # Euclidean distance between the two positions
    if r == 0:
        raise ValueError("Collision or zero distance between objects.")
    return G * m2 / r**3 * r_vec  # Gravitational acceleration vector pointing toward the other mass

# Define the Verlet step for gravity
def verlet_step_gravity(x, v, m_others, x_others, dt, G):
    a = np.sum([gravitational_acceleration(m_other, x, x_other, G) for x_other, m_other in zip(x_others, m_others)], axis=0)
    # print(a)
    x_new = x + v * dt + 0.5 * a * dt**2
    a_new = np.sum([gravitational_acceleration(m_other, x_new, x_other, G) for x_other, m_other in zip(x_others, m_others)], axis=0)
    v_new = v + 0.5 * (a + a_new) * dt
    return x_new, v_new

# Define the update function for all objects
def verlet_update_all(x, v, m, dt, G, update=True):
    n_objects = x.shape[0]
  
    if isinstance(update, bool):
        update = [update] * n_objects
    if isinstance(dt, (int, float)):
        dt = [dt] * n_objects
    
    x_new = np.copy(x)
    v_new = np.copy(v)
    for i in range(n_objects):
        x_others = np.vstack([x[j] for j in range(n_objects) if j != i])
        m_others = np.array([m[j] for j in range(n_objects) if j != i])
        if update[i]:
            x_new[i], v_new[i] = verlet_step_gravity(x[i], v[i], m_others, x_others, dt[i], G)
    x[:], v[:] = x_new, v_new

