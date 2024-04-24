
    
   # Import everything from question 1
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

    v = np.array([[0, 0], [0, velocity]])  # Initial velocities

    total_time = 0
    min_distance = np.inf
    pos=None
    left_original_position = False
    areas=[]
    times=[]
    while True:
        old_x=np.copy(x)
        verlet_update_all(x, v, m, dt, G, [False, True])
        distance_moved=np.linalg.norm(x[1]-old_x[1])
        distance_to_sun=np.linalg.norm(x[1])
        areas.append(0.5*distance_moved*distance_to_sun )
        times.append(total_time)
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
    
    print(f"Total time taken to complete one orbit is {total_time / (60 * 60 * 24)} days, pos:{pos}")
    plt.figure(figsize=(8, 6))  # Size of the plot
    plt.plot(times, areas, label='areas at time')  # Plot the data
    
    plt.title('Plot')  # Title of the plot
    plt.xlabel('area')  # X-axis label
    plt.ylabel('time')  # Y-axis label
    plt.show()
    
    plt.figure(figsize=(8, 6))  # Size of the plot
    plt.hist(areas, bins=30, label='Areas Distribution')  # Plot the histogram of 'areas'

    plt.title('Histogram of Areas')  # Title of the plot
    plt.xlabel('Value')  # X-axis label
    plt.ylabel('Frequency')  # Y-axis label
    plt.legend()  # Display legend
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
    
   