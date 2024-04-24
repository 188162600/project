#Importing everything from question 1
from core1 import *

def main():
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
    min_distance_from_sun = np.inf
    max_distance_form_sun = 0
    min_distance_from_sun_v=0
    max_distance_form_sun_v=0

    v = np.array([[0, 0], [0, velocity]])  # Initial velocities

    total_time = 0
    min_distance = np.inf
    pos=None
    left_original_position = False
    
    while True:
        verlet_update_all(x, v, m, dt, G, [False, True])
        distance_from_starting_position = np.linalg.norm(x[1] - earth_starting_position)
        
        if distance_from_starting_position < min_distance and left_original_position:
            min_distance = distance_from_starting_position
            pos=x[1]
            print(min_distance, "meters", total_time / (60 * 60 * 24), "days", pos)
        distance_from_sun=np.linalg.norm(x[1])
        if distance_from_sun < min_distance_from_sun:
            min_distance_from_sun=distance_from_sun
            min_distance_from_sun_v=np.linalg.norm(v[1])
        if distance_from_sun > max_distance_form_sun:
            max_distance_form_sun=distance_from_sun
            max_distance_form_sun_v=np.linalg.norm(v[1])
            # print(max_distance_form_sun_v,"v")
        
        if distance_from_starting_position < precision:
            
            if left_original_position:
                break
        else:
            left_original_position = True
        
        total_time += dt
    a=(max_distance_form_sun-min_distance_from_sun)/2
    b=np.sqrt(max_distance_form_sun*min_distance_from_sun)
    epsilon=(max_distance_form_sun-min_distance_from_sun)/(max_distance_form_sun+min_distance_from_sun)
    escape_vel=escape_velocity(m[0],max_distance_form_sun,G)
    print(f"Total time taken to complete one orbit is {total_time / (60 * 60 * 24)} days, pos:{pos}")
    print(f"Min distance from sun is {min_distance_from_sun} meters. v={min_distance_from_sun_v}, energy={energy(m,min_distance_from_sun_v)}",)
    print(f"Max distance from sun is {max_distance_form_sun} meters. v={max_distance_form_sun_v}, energy={energy(m,max_distance_form_sun_v)}")
   
    print(f"a={a}, b={b}, epsilon={epsilon}")
    print(f"Escape velocity at max distance from sun is {escape_vel}")
    

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
    
   #Importing everything from question 1
from core1 import *

def main():
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
    min_distance_from_sun = np.inf
    max_distance_form_sun = 0
    min_distance_from_sun_v=0
    max_distance_form_sun_v=0

    v = np.array([[0, 0], [0, velocity]])  # Initial velocities

    total_time = 0
    min_distance = np.inf
    pos=None
    left_original_position = False
    
    while True:
        verlet_update_all(x, v, m, dt, G, [False, True])
        distance_from_starting_position = np.linalg.norm(x[1] - earth_starting_position)
        
        if distance_from_starting_position < min_distance and left_original_position:
            min_distance = distance_from_starting_position
            pos=x[1]
            print(min_distance, "meters", total_time / (60 * 60 * 24), "days", pos)
        distance_from_sun=np.linalg.norm(x[1])
        if distance_from_sun < min_distance_from_sun:
            min_distance_from_sun=distance_from_sun
            min_distance_from_sun_v=np.linalg.norm(v[1])
        if distance_from_sun > max_distance_form_sun:
            max_distance_form_sun=distance_from_sun
            max_distance_form_sun_v=np.linalg.norm(v[1])
            # print(max_distance_form_sun_v,"v")
        
        if distance_from_starting_position < precision:
            
            if left_original_position:
                break
        else:
            left_original_position = True
        
        total_time += dt
    a=(max_distance_form_sun-min_distance_from_sun)/2
    b=np.sqrt(max_distance_form_sun*min_distance_from_sun)
    epsilon=(max_distance_form_sun-min_distance_from_sun)/(max_distance_form_sun+min_distance_from_sun)
    escape_vel=escape_velocity(m[0],max_distance_form_sun,G)
    print(f"Total time taken to complete one orbit is {total_time / (60 * 60 * 24)} days, pos:{pos}")
    print(f"Min distance from sun is {min_distance_from_sun} meters. v={min_distance_from_sun_v}, energy={energy(m,min_distance_from_sun_v)}",)
    print(f"Max distance from sun is {max_distance_form_sun} meters. v={max_distance_form_sun_v}, energy={energy(m,max_distance_form_sun_v)}")
   
    print(f"a={a}, b={b}, epsilon={epsilon}")
    print(f"Escape velocity at max distance from sun is {escape_vel}")
    

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
    
   