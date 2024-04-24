from core1 import *
def fix_velocity_constant(v,speed):
    direction=v/np.linalg.norm(v)
    return direction*speed
def main():
    is_speed_constant=True
    # Constants and initial conditions
    G = 6.67430e-11  # Gravitational constant
    dt = 60 * 60/10  # Time step (1 hour)
    
    # Masses of Sun and Earth, positions, and velocities
    m = np.array([1.989e30, 5.972e24])  # Masses in kg
    distance = 149.6e9  # Distance in meters (1 AU)
    earth_starting_position = np.array([distance, 0])
    
    x = np.array([[0, 0], [distance, 0]])  # Initial positions

    # Orbital velocity (m/s)
    velocity = 29785.189029332814
  
    precision = velocity * dt * 2
    print("Precision:", precision, "meters")

    v = np.array([[0, 0], [0, velocity]])  # Initial velocities

    total_time = 0
    min_distance = np.inf
    pos=None
    left_original_position = False
    kinetic_energies=[]
    potential_energies=[]
    total_energies=[]
    xs=[]
    while True:
        verlet_update_all(x, v, m, dt, G, [False, True])
        xs.append(x.copy())
        kinetic_energies.append(np.linalg.norm(kinetic_energy(m[1],np.linalg.norm(v[1]))))
        potential_energies.append(np.linalg.norm(potential_energy(m[0],m[1],np.linalg.norm(x[1]-x[0]),G)))
        total_energies.append(kinetic_energies[-1]+potential_energies[-1])
        distance_from_starting_position = np.linalg.norm(x[1] - earth_starting_position)
        
        if distance_from_starting_position < min_distance and left_original_position:
            min_distance = distance_from_starting_position
            pos=x[1]
            print(min_distance, "meters", total_time / (60 * 60 * 24), "days", pos)
        
        if distance_from_starting_position < precision:
            
            if left_original_position:
                break
        else:
            left_original_position = True
        
        total_time += dt
    radius = np.pi * distance * 2  # Orbit circumference
    analytically_velocity = radius /total_time  # Orbital velocity (m/s)
    xs=np.array(xs)
    print(f"Total time taken to return to origin is {total_time / (60 * 60 * 24)} days, pos:{pos} and therefore orbit is closed")
    print(f"Analytically calculated velocity is {analytically_velocity} m/s given r{distance},t{total_time/ (60 * 60 * 24)} days")
    

    print(f"Energy is conserved as the diff{np.mean(np.abs( total_energies-np.mean(total_energies)))/np.mean(total_energies)}")
   
    plt.figure(figsize=(10,5))
    plt.plot(xs[:,1,0],xs[:,1,1],label="Earth")
    plt.plot(xs[:,0,0],xs[:,0,1],label="Sun")
    plt.title("Closed orbit of Earth around Sun")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    
    plt.figure(figsize=(10,5))
    plt.plot(kinetic_energies,label="Kinetic Energy")
    plt.plot(potential_energies,label="Potential Energy")
    plt.plot(total_energies,label="Total Energy")
    print(max(kinetic_energies),max(potential_energies),max(total_energies),min(kinetic_energies),min(potential_energies),min(total_energies))
    plt.title("Conserved Total energy of Earth")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()
    

    # # Plot the trajectories
    # plt.figure(figsize=(10, 5))
    # plt.plot(positions[:, 0, 0], positions[:, 0, 1], 'o-', label='Object 1')
    # plt.plot(positions[:, 1, 0], positions[:, 1, 1], 'o-', label='Object 2')
    # plt.title("Trajectories of Two Objects")
    # plt.xlabel("X Position (meters)")
    # plt.ylabel("Y Position (meters)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__=="__main__":
    main()
    
   