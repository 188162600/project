import numpy as np
import matplotlib.pyplot as plt

# # Define the gravitational acceleration function
# def gravitational_acceleration(m2, x1, x2, G=6.67430e-11):
#     r_vec = x2 - x1  # displacement vector from m1 to m2
#     r = np.linalg.norm(r_vec)  # Euclidean distance between m1 and m2
#     if r == 0:
#         raise ValueError("Collision or zero distance between objects.")
#     return G * m2 / r**3 * r_vec  # Gravitational acceleration vector

def kinetic_energy(m, v): #Defining function to help to determine the kinetic energy
    """
    Calculate the kinetic energy of an object.

    Parameters:
    m (float): Mass of the object (in kilograms).
    v (float): Velocity of the object (in meters per second).

    Returns:
    float: Kinetic energy of the object.
    """
    return 0.5 * m * v**2   #Returns kinetic energy equation 1/2mv^2
def potential_energy(m1, m2, r, G): #Defining function to help to determine the potential energy
    """
    Calculate the gravitational potential energy between two objects.

    Parameters:
    m1 (float): Mass of the first object (in kilograms).
    m2 (float): Mass of the second object (in kilograms).
    r (float): Distance between the two objects (in meters).
    G (float): Gravitational constant (in m^3 kg^-1 s^-2).

    Returns:
    float: Gravitational potential energy between the two objects.
    """
    return -G * m1 * m2 / r    

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

def main():
         # Constants and initial conditions
    G = 6.67430e-11  # Gravitational constant
    dt = 3600  # Time step (1 hour)
    # total_time = 24 * 25
    period=24*27.32*60*60
    total_time=period*1
    times = np.arange(0, total_time, dt)

    # Masses, initial positions (x, y) and velocities (vx, vy) for two objects
    m = np.array([5.97e24, 7.35e22])  # Masses of Earth and Moon
    x = np.array([[0, 0], [384400e3, 0]])  # Initial positions
    distance=384400e3
    radius=np.pi*distance*2
    velocity=radius/period

   

    # print(velocity)
    v = np.array([[0, 0], [0, velocity]])  # Initial velocities
    velocities=np.zeros((len(times), 2))  
    speeds=np.zeros(len(times))
    
    # Simulation loop
    positions = np.zeros((len(times), 2, 2))  # To store positions for plotting
    for i in range(len(times)):
        positions[i] = x
        velocities[i]= v[1]
        speeds[i]=np.linalg.norm(velocities[i])
        
        verlet_update_all(x, v, m, dt, G,[False,True])
        
    radian_in_time=2*np.pi* times/total_time
    theoretical_x=(np.cos(radian_in_time)*distance,np.sin(radian_in_time)*distance)
    theoretical_v=(np.sin(radian_in_time)*velocity,np.cos(radian_in_time)*velocity)
    theoretical_speed=np.zeros(len(times))
    theoretical_speed[:]=velocity
    # (v,x)
    # plt.figure(figsize=(10,5))
    # plt.plot(times,velocities[])
    # Plot the trajectories
    plt.figure(figsize=(10, 5))
   
    plt.plot(times, positions[:, 1, 0], 'o-', label='Earth x')
    plt.plot(times, positions[:, 1, 1], 'o-', label='Earth y')
    plt.plot(times,theoretical_x[0], 'o-', label='theoretical_x')
    plt.plot(times,theoretical_x[1],'o-', label='theoretical_y')
    
    plt.title("Trajectories of Two Objects")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10,5))
    # print(velocities.shape,times.shape)
    plt.plot(positions[:, 1, 0],velocities[:,0],label="velocity x")
    plt.plot( positions[:, 1, 1],velocities[:,1],label="velocity y")
    
    
    
    plt.plot(theoretical_x[0],theoretical_v[0],label="theoretical velocity x")
    plt.plot(theoretical_x[1],theoretical_v[1],label="theoretical velocity y")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__=="__main__":
    main()
