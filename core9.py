import numpy as np
import matplotlib.pyplot as plt
from core1 import verlet_update_all
import tqdm
import datetime
import threading
import multiprocessing
import copy
from datetime import timezone
import itertools


import datetime

from zoneinfo import ZoneInfo
try:
    from skyfield.api import load
except ImportError:
    import pip
    pip.main(['install', 'skyfield'])
# Load data

#We will use this to fetch 20 points one for each year

def get(t: datetime.datetime):
    # Load planetary data
    planets = load('de421.bsp')
    
    # Convert the provided datetime to a Skyfield time object
    ts = load.timescale()
    datet=t
    t = ts.from_datetime(t)
    
    # Get the celestial bodies
    earth = planets['earth barycenter']
    moon = planets['moon']
    sun = planets['sun']
    
    # Position data at the specified time
    earth_position = earth.at(t)
    moon_position = moon.at(t)
    sun_position = sun.at(t)
    # earth.mass
    # Calculate position relative to the Sun
    pos_earth_rel_sun = earth_position.position.m - sun_position.position.m
    pos_moon_rel_sun = moon_position.position.m - sun_position.position.m

    # Calculate velocities relative to the Sun
    vel_earth_rel_sun = earth_position.velocity.m_per_s - sun_position.velocity.m_per_s
    vel_moon_rel_sun = moon_position.velocity.m_per_s - sun_position.velocity.m_per_s

    # Print positions and velocities
    # print("Position of Earth relative to Sun (AU):", pos_earth_rel_sun)
    # print("Position of Moon relative to Sun (AU):", pos_moon_rel_sun)
    # print("Velocity of Earth relative to Sun (AU/day):", vel_earth_rel_sun)
    # print("Velocity of Moon relative to Sun (AU/day):", vel_moon_rel_sun)
    # print(f"{datet}:{np.array([[0,0,0],pos_earth_rel_sun, pos_moon_rel_sun]), np.array([[0,0,0],vel_earth_rel_sun, vel_moon_rel_sun])},")
    
    return np.array([[0,0,0],pos_earth_rel_sun, pos_moon_rel_sun]), np.array([[0,0,0],vel_earth_rel_sun, vel_moon_rel_sun])



correct_eclipses  = [
    (datetime.datetime(2010, 1, 15, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2010, 7, 11, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2011, 1, 4, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2011, 6, 1, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2011, 7, 1, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2011, 11, 25, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2012, 5, 20, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2012, 11, 13, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2013, 5, 10, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2013, 11, 3, tzinfo=timezone.utc), "Hybrid"),
    (datetime.datetime(2014, 4, 29, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2014, 10, 23, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2015, 3, 20, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2015, 9, 13, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2016, 3, 9, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2016, 9, 1, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2017, 2, 26, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2017, 8, 21, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2018, 2, 15, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2018, 7, 13, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2018, 8, 11, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2019, 1, 6, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2019, 7, 2, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2019, 12, 26, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2020, 6, 21, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2020, 12, 14, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2021, 6, 10, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2021, 12, 4, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2022, 4, 30, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2022, 10, 25, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2023, 4, 20, tzinfo=timezone.utc), "Hybrid"),
    (datetime.datetime(2023, 10, 14, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2024, 4, 8, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2024, 10, 2, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2025, 3, 29, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2025, 9, 21, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2026, 2, 17, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2026, 8, 12, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2027, 2, 6, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2027, 8, 2, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2028, 1, 26, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2028, 7, 22, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2029, 1, 14, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2029, 6, 12, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2029, 7, 11, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2029, 12, 5, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2030, 5, 1, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2030, 11, 25, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2031, 4, 21, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2031, 10, 14, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2032, 3, 9, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2032, 9, 2, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2033, 2, 25, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2033, 7, 23, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2033, 8, 21, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2033, 12, 17, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2034, 5, 14, tzinfo=timezone.utc), "Annular"),
    (datetime.datetime(2034, 11, 7, tzinfo=timezone.utc), "Total"),
    (datetime.datetime(2035, 5, 3, tzinfo=timezone.utc), "Partial"),
    (datetime.datetime(2035, 10, 29, tzinfo=timezone.utc), "Partial"),
]


def project_to_plane(p1, p2, p3):
    """
    Projects three non-collinear points in 3D onto a 2D plane defined by them.
    
    Args:
    p1, p2, p3 (numpy.array): 3D coordinates of the points as numpy arrays.
    
    Returns:
    tuple of numpy.array: 2D coordinates of the three points on the new plane.
    """
    # Calculate vectors from p1 to p2 and p1 to p3
    u = p2 - p1
    v = p3 - p1

    # Calculate the normal vector to the plane using the cross product
    n = np.cross(u, v)

    # Create orthogonal unit vectors in the plane
    u_norm = u / np.linalg.norm(u)
    v_perp = np.cross(n, u)
    v_norm = v_perp / np.linalg.norm(v_perp)

    # Define a function to project a point onto the plane
    def project_point(p):
        # Convert the point's position to the plane's basis
        return np.array([(p - p1) @ u_norm, (p - p1) @ v_norm])

    # Project each point to the plane
    return (project_point(p1), project_point(p2), project_point(p3))

def does_line_intersect_circle(r, center, point, direction):
    # Define the components of the center, point, and direction
    h, k = center
    x0, y0 = point
    dx, dy = direction

    # Compute the coefficients of the quadratic equation (At^2 + Bt + C = 0)
    A = dx**2 + dy**2
    B = 2 * (dx * (x0 - h) + dy * (y0 - k))
    C = (x0 - h)**2 + (y0 - k)**2 - r**2

    # Calculate the discriminant to determine the nature of the roots
    discriminant = B**2 - 4 * A * C

    # If the discriminant is negative, there are no real roots, and the line does not intersect the circle
    if discriminant < 0:
        return False
    else:
        return True
def get_intersection(p1, direction1, p2, direction2):
    # Unpack points and direction vectors
    x1, y1 = p1
    dx1, dy1 = direction1
    x2, y2 = p2
    dx2, dy2 = direction2

    # Check if the direction vectors are parallel (cross product is zero)
    if dx1 * dy2 == dy1 * dx2:
        return None  # No intersection (parallel lines)

    # Solve the system of equations:
    # x1 + t*dx1 = x2 + s*dx2
    # y1 + t*dy1 = y2 + s*dy2
    # Using matrix representation: A * [t; s] = B
    # Where A = [[dx1, -dx2], [dy1, -dy2]] and B = [x2 - x1, y2 - y1]
    # The solution for t is given by t = det(B, A_col2) / det(A)
    det_A = dx1 * -dy2 - (-dx2) * dy1
    det_t = (x2 - x1) * -dy2 - (y2 - y1) * -dx2
    t = det_t / det_A

    # Calculate the intersection point using t
    intersection_x = x1 + t * dx1
    intersection_y = y1 + t * dy1

    return (intersection_x, intersection_y)
def line_intersection(p1, direction1, p2, direction2):
    """Finds the intersection point of two lines given by points and direction vectors."""
    A = np.array([direction1, -direction2]).T
    b = np.array(p2) - np.array(p1)
    try:
        t, s = np.linalg.solve(A, b)
        return np.array(p1) + t * np.array(direction1)
    except np.linalg.LinAlgError:
        return None  # Lines are parallel

def line_intersection(p1, d1, p2, d2):
    """Calculate the intersection of two lines given by points and direction vectors."""
    A = np.array([d1, -d2]).T
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel
    b = np.array(p2) - np.array(p1)
    t = np.linalg.solve(A, b)
    return np.array(p1) + t[0] * d1


def does_line_intersect_before_circle(r, center, p1, direction1, p2, direction2):
    def line_intersection(p1, d1, p2, d2):
        # Convert points and directions into numpy arrays for vectorized computation
        p1, d1, p2, d2 = map(np.array, (p1, d1, p2, d2))
        # Matrix to solve for intersection
        A = np.array([d1, -d2]).T
        if np.linalg.det(A) == 0:
            return None  # Lines are parallel; no intersection
        # Solve linear system
        t = np.linalg.solve(A, p2 - p1)
        return p1 + t[0] * d1

    def line_circle_intersection(p, d, center, r):
        # Move circle center to origin
        p = np.array(p) - np.array(center)
        d = np.array(d)
        a = np.dot(d, d)
        b = 2 * np.dot(p, d)
        c = np.dot(p, p) - r**2
        delta = b**2 - 4*a*c
        if delta < 0:
            return []  # No intersection
        elif delta == 0:
            t = -b / (2*a)
            return [np.array(center) + (np.array(p) + t * np.array(d))]
        else:
            sqrt_delta = np.sqrt(delta)
            t1 = (-b + sqrt_delta) / (2*a)
            t2 = (-b - sqrt_delta) / (2*a)
            return [np.array(center) + (np.array(p) + t1 * np.array(d)),
                    np.array(center) + (np.array(p) + t2 * np.array(d))]

    intersection = line_intersection(p1, direction1, p2, direction2)
    if intersection is None:
        return False  # No intersection of lines

    # Calculate distances from line starts to the intersection point
    dist1 = np.linalg.norm(np.array(p1) - np.array(intersection))
    dist2 = np.linalg.norm(np.array(p2) - np.array(intersection))

    # Find intersection points with the circle
    intersects1 = line_circle_intersection(p1, direction1, center, r)
    intersects2 = line_circle_intersection(p2, direction2, center, r)

    # Check if any intersection with the circle occurs before the line intersection
    for point in intersects1:
        if np.linalg.norm(np.array(p1) - point) < dist1:
            return False
    for point in intersects2:
        if np.linalg.norm(np.array(p2) - point) < dist2:
            return False

    return True
def does_line_intersect_half_circle(r, center, norm, point, direction):
    # Extract components for clarity
    h, k = center
    px, py = point
    dx, dy = direction
    
    # Components for the quadratic equation A*t^2 + B*t + C = 0
    A = dx**2 + dy**2
    B = 2 * (dx * (px - h) + dy * (py - k))
    C = (px - h)**2 + (py - k)**2 - r**2
    
    # Discriminant of the quadratic equation
    D = B**2 - 4 * A * C
    
    # If the discriminant is negative, no real intersection
    if D < 0:
        return False
    else:
        # Calculate potential t values for intersections
        t1 = (-B + np.sqrt(D)) / (2 * A)
        t2 = (-B - np.sqrt(D)) / (2 * A)
        
        # Check if these t values are within the half-circle bounds using the normal vector
        intersect_points = []
        for t in (t1, t2):
            x = px + t * dx
            y = py + t * dy
            if np.dot(norm, [x - h, y - k]) >= 0:  # Checking the half-plane condition
                intersect_points.append((x, y))
        
        # If we have at least one valid intersection point
        return len(intersect_points) > 0
def radian_between_vectors(v1,v2):
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

def is_solar_eclipse_aligned(sun_pos, moon_pos, earth_pos, ep=None):
    """
    Determines if the conditions are right for a solar eclipse, with the Moon
    positioned between the Sun and the Earth.

    Parameters:
    - sun_pos: Position vector of the Sun.
    - moon_pos: Position vector of the Moon.
    - earth_pos: Position vector of the Earth.
    - ep: Epsilon, the tolerance level for alignment.

    Returns:
    - True if the conditions for a solar eclipse are met, False otherwise.
    """
    # Calculate unit vectors
    if ep is None:
        ep=0.001
    sun_to_moon = moon_pos - sun_pos
    sun_to_moon_unit = sun_to_moon / np.linalg.norm(sun_to_moon)  # Normalize vector

    sun_to_earth = earth_pos - sun_pos
    earth_to_moon = moon_pos - earth_pos
    earth_to_moon_unit = earth_to_moon / np.linalg.norm(earth_to_moon)  # Normalize vector

    # Check for alignment using dot product to ensure Moon is between Sun and Earth
    # This checks that the angle between sun_to_moon and earth_to_moon is approximately 180 degrees
    angle_cosine = np.dot(sun_to_moon_unit, earth_to_moon_unit)
    
    # np.allclose to check if angle_cosine is close to -1 (cos(180 degrees) = -1), within the specified tolerance
    moon_is_closer_to_sun =np.linalg.norm(sun_to_moon)< np.linalg.norm(sun_to_earth)  
    return moon_is_closer_to_sun and np.isclose(angle_cosine, -1, atol=ep) ,abs(angle_cosine-(-1))

def is_solar_eclipse(sun_pos, moon_pos, earth_pos, sun_radius, moon_radius, earth_radius,ep=None,ep_hybrid=None):
  
    align,gap=is_solar_eclipse_aligned(sun_pos, moon_pos, earth_pos,ep)
    if not align:
    # if not is_solar_eclipse_aligned(sun_pos, moon_pos, earth_pos,ep):
        return False, "none",None
    
    sun_pos,moon_pos,earth_pos=project_to_plane(sun_pos,moon_pos,earth_pos)
    
    sun_to_moon = moon_pos - sun_pos
    sun_to_earth = earth_pos - sun_pos
    earth_to_sun = sun_pos - earth_pos
   
     
    # if np.linalg.norm(sun_to_earth) <np.linalg.norm(sun_to_moon):
    #     return False, "none"
                                                                                                                                                                                                         


    sun_top = sun_pos +np.array( [0,1])* sun_radius
    sun_bot = sun_pos -np.array( [0,1])* sun_radius
    moon_top = moon_pos +np.array( [0,1])* moon_radius
    moon_bot = moon_pos -np.array( [0,1])* moon_radius
    

    penumbra_top_start = sun_top
    penumbra_top_direction = moon_bot - sun_top
    penumbra_bot_start = sun_bot
    penumbra_bot_direction = moon_top - sun_bot

   
    is_penumbra_top_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos, earth_to_sun,penumbra_top_start, penumbra_top_direction)
    is_penumbra_bot_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos,earth_to_sun, penumbra_bot_start, penumbra_bot_direction)

    umbra_top_start = sun_top
    umbra_top_direction = moon_top - sun_top
    umbra_bot_start = sun_bot
    umbra_bot_direction = moon_bot - sun_bot
    
  
    is_umbra_top_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos,earth_to_sun,umbra_top_start, umbra_top_direction)
    is_umbra_bot_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos,earth_to_sun,umbra_bot_start, umbra_bot_direction)
    
    is_partial_eclipse = is_penumbra_top_in_earth or is_penumbra_bot_in_earth 
    # is_annular_eclipse = is_penumbra_top_in_earth and is_penumbra_bot_in_earth
    is_total_or_annular = is_umbra_top_in_earth or is_umbra_bot_in_earth
    # if is_total_or_annular and is_partial_eclipse:
    #     return True, "Hybrid", gap
    if is_total_or_annular:
        # if is_partial_eclipse:
        #     return True, "Hybrid", gap
       
        # line_intersect_before_half_circle=np.linalg.norm(line_intersection-earth_pos)<earth_radius  and np.linalg.norm(line_intersection-sun_pos)<np.linalg.norm(sun_to_earth)
        
        line_intersect_before_half_circle=does_line_intersect_before_circle(earth_radius, earth_pos,umbra_top_start, umbra_top_direction, umbra_bot_start, umbra_bot_direction)
        # line_intersect_after_half_circle= does_line_intersect_before_circle(earth_radius, earth_pos,umbra_top_start, -umbra_top_direction, umbra_bot_start, -umbra_bot_direction)
        # if line_intersect_before_half_circle and line_intersect_after_half_circle:
        #     return True, "Hybrid", gap
        return  True, "Annular" if line_intersect_before_half_circle else "Total", gap
    
   
    if is_partial_eclipse:
        return  True, "Partial", gap
    return False, "None",None
    
    
    
def find_closest_eclipse(eclipses, date,days_limit=float("inf")):
    closest_eclipse = None
    closest_distance = float("inf")
    for eclipse in eclipses:
        distance = abs((eclipse[0] - date).days)
        if distance < closest_distance and distance<days_limit:
            closest_distance = distance
            closest_eclipse = eclipse
    return closest_eclipse

  
# def find_all_eclipses(x, v, m, dt, G,r, total_time, start_date):
#     # print("total_time",total_time,dt)
#     times = np.arange(0, total_time, dt)
#     # print(len(times))
#     # eclipse_types=set()
#     current_eclipse_time_times=dict()
#     eclipses=[]
#     xs=[]
#     xs_2d=[]
#     eclipse_start=False
#     # ep=dt/60000
#     # print("times",times.__len__())
#     try:
#         for current_time in tqdm.tqdm( times):
#             xs.append(copy.deepcopy(x))
#             # projected=project_to_plane(x[0],x[1],x[2])
#             # xs_2d.append(projected[1]-projected[0])
#             # print((projected[1]-projected[0]).shape)

#             verlet_update_all(x, v, m, dt, G, [True, True,True])
#             # if not is_solar_eclipse_aligned(x[0], x[2], x[1]):
#             #     continue
#             x_sun, x_earth, x_moon = x
#             result,type,gap=is_solar_eclipse(x_sun, x_moon, x_earth, r[0], r[2], r[1])
#             # eclipse_types.add(type)
#             if result:
#                 eclipse_start = True
#                 # eclipse_start_time= current_time
#                 current_eclipse_time_times.update({gap:(type,current_time)})
#             else:
#                 if eclipse_start:
#                     eclipse_start = False
#                     # print(start_date)
                
#                     # if is_solar_eclipse_aligned(x[0], x[2], x[1]):
#                     # print("gap",gap)
#                     # current_eclipse_time_times.update({gap:current_time})
#                     info=min(current_eclipse_time_times.items(), key=lambda x: x[0])[1]
#                     center_time=info[1]
#                     eclipse_types=info[0]
#                     center_time=start_date+datetime.timedelta(seconds=int(center_time))
#                     eclipses.append((center_time,eclipse_types))
#                     # eclipse_types=set()

#     except Exception as e:
#         print(e)
#         pass
    return eclipses
  
def find_all_eclipses_accurate(x, v, m, dt, G,r, total_time, start_date):
    # print("total_time",total_time,dt)
    times = np.arange(0, total_time, dt)
    # print(len(times))
    # eclipse_types=set()
    current_eclipse_time_times=dict()
    eclipses=[]
 
    eclipse_start=False
    # ep=dt/60000
    # print("times",times.__len__())
    try:
        for current_time in tqdm.tqdm( times):
            
            # projected=project_to_plane(x[0],x[1],x[2])
            # xs_2d.append(projected[1]-projected[0])
            # print((projected[1]-projected[0]).shape)
            x=get(start_date+datetime.timedelta(seconds=int(current_time)))[0]
            # verlet_update_all(x, v, m, dt, G, [True, True,True])
            # if not is_solar_eclipse_aligned(x[0], x[2], x[1]):
            #     continue
            x_sun, x_earth, x_moon = x
            result,type,gap=is_solar_eclipse(x_sun, x_moon, x_earth, r[0], r[2], r[1])
            # eclipse_types.add(type)
            if result:
                eclipse_start = True
                # eclipse_start_time= current_time
                current_eclipse_time_times.update({gap:(type,current_time)})
            else:
                if eclipse_start:
                    eclipse_start = False
                    # print(start_date)
                
                    # if is_solar_eclipse_aligned(x[0], x[2], x[1]):
                    # print("gap",gap)
                    # current_eclipse_time_times.update({gap:current_time})
                    info=min(current_eclipse_time_times.items(), key=lambda x: x[0])[1]
                    center_time=info[1]
                    eclipse_types=info[0]
                    center_time=start_date+datetime.timedelta(seconds=int(center_time))
                    eclipses.append((center_time,eclipse_types))
                    # eclipse_types=set()

    except Exception as e:
        print(e)
        pass
    return eclipses

  
def find_all_eclipses_speed(x, v, m, dt, G,r, total_time, start_date):
    # print("total_time",total_time,dt)
    times = np.arange(0, total_time, dt)
    # print(len(times))
    # eclipse_types=set()
    current_eclipse_time_times=dict()
    eclipses=[]
 
    eclipse_start=False
    # ep=dt/60000
    # print("times",times.__len__())
    try:
        for current_time in tqdm.tqdm( times):
            
            # projected=project_to_plane(x[0],x[1],x[2])
            # xs_2d.append(projected[1]-projected[0])
            # print((projected[1]-projected[0]).shape)
            
            verlet_update_all(x, v, m, dt, G, [True, True,True])
            # if not is_solar_eclipse_aligned(x[0], x[2], x[1]):
            #     continue
            x_sun, x_earth, x_moon = x
            result,type,gap=is_solar_eclipse(x_sun, x_moon, x_earth, r[0], r[2], r[1])
            # eclipse_types.add(type)
            if result:
                eclipse_start = True
                # eclipse_start_time= current_time
                current_eclipse_time_times.update({gap:(type,current_time)})
            else:
                if eclipse_start:
                    eclipse_start = False
                    # print(start_date)
                
                    # if is_solar_eclipse_aligned(x[0], x[2], x[1]):
                    # print("gap",gap)
                    # current_eclipse_time_times.update({gap:current_time})
                    info=min(current_eclipse_time_times.items(), key=lambda x: x[0])[1]
                    center_time=info[1]
                    eclipse_types=info[0]
                    center_time=start_date+datetime.timedelta(seconds=int(center_time))
                    eclipses.append((center_time,eclipse_types))
                    # eclipse_types=set()

    except Exception as e:
        print(e)
        pass
    return eclipses
def find_all_eclipses_speed_args(args):
    return find_all_eclipses_speed(*args)
def find_all_eclipses_accurate_args(args):
    return find_all_eclipses_accurate(*args)
# def find_all_eclipses2(args):
#     x, v, m, dt, G,r, total_time,start_date=args
#     # print("total_time",total_time)
#     return find_all_eclipses(x, v, m, dt, G,r, total_time, start_date)


def split_array(times, n):
    k, m = divmod(len(times), n)
    return (times[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def separate_orbit(x, v, m, dt, G,  start_date, total_time,n):
    xs = [x.copy()]
    vs=[v.copy()]
    times=np.arange(0, total_time, dt)
    times=list(split_array(times,n))
    # print(times)
    # print(total_time//n//dt)
    dates=[start_date]
    
    for time in tqdm.tqdm( times,desc="Splitting"):
        for current_time in tqdm.tqdm( time,desc="Generating"):
            verlet_update_all(x, v, m, dt, G, [True, True,True])
        
        dates.append(start_date+datetime.timedelta(seconds=int(current_time)))
        xs.append(x.copy())
        vs.append(v.copy())
    del xs[-1]
    del vs[-1]
   
    return xs,vs,int(total_time//n)



        
    

        
        
    


    
def print_eclipses(eclipses,correct_eclipses,start_date,total_time):
    # print(correct_eclipses)
    diffs=[]
    unmatched_eclipses = set(correct_eclipses)
    for eclipse_time, eclipse_type in eclipses:
      
        closest_correct_eclipse = find_closest_eclipse(correct_eclipses, eclipse_time)
        unmatched_eclipses.discard(closest_correct_eclipse)
        day_diff=(eclipse_time-closest_correct_eclipse[0]).days
        print(f"At {eclipse_time}, {eclipse_type} occurred, closest correct eclipse{closest_correct_eclipse}. Time Difference:{day_diff} days")
        diffs.append(day_diff)
        
    end_date=start_date+datetime.timedelta(seconds=total_time)
    print(f"Starting Time:{start_date}, Ending Time:{end_date}")
        # print(len(eclipses ))
    unmatched_eclipses=list(filter(lambda x:x[0]>start_date and x[0]<end_date,unmatched_eclipses))
    
    plt.hist(diffs, edgecolor='black', align='left')
    plt.xlabel("Difference in Days")
    plt.ylabel("Frequency")
    plt.title("Difference in Days between Correct and Found Eclipses")
    
    plt.show()
    
    

def main_speed():
    # print(correct_eclipses)
    m = np.array([1.9885e30, 5.97237e24, 7.342e22])

    r = np.array([696340e3, 6371e3, 1737.4e3])
    dt=600

    
    # dt=60
    G=6.67430e-11
   
   
    start_date=datetime.datetime(2010, 6, 21, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
    workers=multiprocessing.cpu_count()
    splits=2000
    total_time=3600*24*365*15
    times=np.arange(0,total_time,dt)
    spited_times=list(split_array(times,splits))
    date=[start_date+datetime.timedelta(seconds=int(spited_times[i][0])) for i in range(splits)]
    # print(date)
    secs_per_worker=[total_time//splits for i in range(splits)]

    xs=[]
    vs=[]
    for i in range(splits):
        xs.append(get(start_date+datetime.timedelta(seconds=int( spited_times[i][0])))[0])
        vs.append(get(start_date+datetime.timedelta(seconds= int(spited_times[i][0])))[1])
        
    
  
    # print(secs_per_year,date)
   
    args=[(xs[i],vs[i],m,dt,G,r,secs_per_worker[i],date[i]) for i in range(splits)]

    pool = multiprocessing.Pool(processes=workers)
    # find_all_eclipses()
    # Map `square_number` to the numbers
    results = pool.map(find_all_eclipses_speed_args, args)

    pool.close()
    pool.join()
    print("Eclipses Found")
    # print(results)
    results=list(itertools.chain(*results))
    # print(results)
    print_eclipses(results,correct_eclipses,start_date,total_time)
    # while threading.active_count()>1:
    #     pass
    # print("Eclipses Found")
    # print_eclipses(eclipses,correct_eclipses,start_date,total_time)
def main_accurate():
    m = np.array([1.9885e30, 5.97237e24, 7.342e22])

    r = np.array([696340e3, 6371e3, 1737.4e3])
    dt=600

    
    # dt=60
    G=6.67430e-11
   
   
    start_date=datetime.datetime(2010, 6, 21, 0, 0, 0, tzinfo=ZoneInfo("UTC"))
    workers=multiprocessing.cpu_count()
    splits=20
    total_time=3600*24*365*15
    times=np.arange(0,total_time,dt)
    spited_times=list(split_array(times,splits))
    date=[start_date+datetime.timedelta(seconds=int(spited_times[i][0])) for i in range(splits)]
    # print(date)
    secs_per_worker=[total_time//splits for i in range(splits)]

    xs=[]
    vs=[]
    for i in range(splits):
        xs.append(get(start_date+datetime.timedelta(seconds=int( spited_times[i][0])))[0])
        vs.append(get(start_date+datetime.timedelta(seconds= int(spited_times[i][0])))[1])
        
    
  
    # print(secs_per_year,date)
   
    args=[(xs[i],vs[i],m,dt,G,r,secs_per_worker[i],date[i]) for i in range(splits)]

    pool = multiprocessing.Pool(processes=workers)
    # find_all_eclipses()
    # Map `square_number` to the numbers
    results = pool.map(find_all_eclipses_accurate_args, args)

    pool.close()
    pool.join()
    print("Eclipses Found")
  
    results=list(itertools.chain(*results))
  
    print_eclipses(results,correct_eclipses,start_date,total_time)
   
def main():
    main_speed()
    # main_accurate()
if __name__ == "__main__":
    main()
    