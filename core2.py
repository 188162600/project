# Import everything from question 1
from core1 import *

#  hhbubunun
def main():
    # Constants and initial conditions
    G = 6.67430e-11  # Gravitational constant
    dt = 60 * 60 / 10  # Time step (1 hour)

    # Masses of Sun and Earth, positions, and velocities
    m = np.array([1.989e30, 5.972e24])  # Masses in kg
    distance = 149.6e9  # Distance in meters (1 AU)
    earth_starting_position = np.array([distance, 0])
    velocity = 29785.189029332814  # Orbital velocity (m/s)
    x_1 = np.array([[0, 0], [distance, 0]])  # Initial positions
    x_2 = np.array([[0, 0], [distance, 0]])
    x_3 = np.array([[0, 0], [distance, 0]])
    v_1 = np.array([[0, 0], [0, velocity]])  # Initial velocity that will create an elipse
    v_2 = np.array([[0, 0], [0, velocity * 2]])  # Initial velocities that will create a hyperbolic trajectorys
    v_3 = np.array([[0, 0], [0, -42127.865427240446]])

    # Defining variables for an eliptical trajectory
    xs_1 = []
    ys_1 = []
    # Defining variables for hyperbolic trajectorys
    xs_2 = []
    ys_2 = []
    xs_3 = []
    ys_3 = []

    precision = velocity * dt * 2  # Setting the precision
    print("Precision:", precision, "meters")

    # Setting inital values for the min and max distances from the sun
    min_distance_from_sun = np.inf
    max_distance_from_sun = 0
    # Defining variables for the min and max velocities
    min_distance_from_sun_v = None
    max_distance_from_sun_v = None

    # Defining more variables for use inside the loop
    total_time = 0
    min_distance = np.inf
    pos = None
    left_original_position = False

    while True:
        # Run simulation acoring to Newton's equations
        xs_1.append(x_1[1][0])
        ys_1.append(x_1[1][1])
        xs_2.append(x_2[1][0])
        ys_2.append(x_2[1][1])
        xs_3.append(x_3[1][0])
        ys_3.append(x_3[1][1])
        verlet_update_all(x_1, v_1, m, dt, G, [False, True])
        verlet_update_all(x_2, v_2, m, dt, G, [False, True])
        verlet_update_all(x_3, v_3, m, dt, G, [False, True])
        distance_from_starting_position = np.linalg.norm(x_1[1] - earth_starting_position)
        # Calculating the minimum and maximum distances (The  from the sun
        if distance_from_starting_position < min_distance and left_original_position:
            min_distance = distance_from_starting_position
            pos = x_1[1]
            print(min_distance, "meters", total_time / (60 * 60 * 24), "days", pos)
        distance_from_sun = np.linalg.norm(x_1[1])
        if distance_from_sun < min_distance_from_sun:
            min_distance_from_sun = distance_from_sun
            min_distance_from_sun_v = np.linalg.norm(v_1[1])
        if distance_from_sun > max_distance_from_sun:
            max_distance_from_sun = distance_from_sun
            max_distance_from_sun_v = np.linalg.norm(v_1[1])
            # print(max_distance_form_sun_v,"v")

        if distance_from_starting_position < precision:

            if left_original_position:
                break
        else:
            left_original_position = True

        total_time += dt
    a = (max_distance_from_sun - min_distance_from_sun) / 2
    b = np.sqrt(max_distance_from_sun * min_distance_from_sun)
    epsilon = (max_distance_from_sun - min_distance_from_sun) / (max_distance_from_sun + min_distance_from_sun)
    escape_vel = escape_velocity(m[0], max_distance_from_sun, G)
    print(f"Total time taken to complete one orbit is {total_time / (60 * 60 * 24)} days, pos:{pos}")
    print(
        f"The minimum distance from sun (P) is {min_distance_from_sun} meters. v={min_distance_from_sun_v}, energy={energy(m, min_distance_from_sun_v)}", )
    print(
        f"The maximum distance from sun (A) is {max_distance_from_sun} meters. v={max_distance_from_sun_v}, energy={energy(m, max_distance_from_sun_v)}")

    print(f"a={a}, b={b}, epsilon={epsilon}")
    print(f"Escape velocity at max distance from sun is {escape_vel}")

    plt.figure(figsize=(10, 5))

    plt.plot(xs_1, ys_1, 'o-', label=f'Object 1, $v_0 = {velocity}$ m/s')
    plt.plot(xs_2, ys_2, 'o-', label=f'Object 2, $v_0 = {velocity * 2}$ m/s')
    plt.plot(xs_3, ys_3, 'o-', label=f'Object 3, $v_0 = {-42128}$ m/s')

    plt.title("Trajectories of Three Objects")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()