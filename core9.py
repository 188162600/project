
    

import numpy as np
import matplotlib.pyplot as plt

import tqdm
import datetime
import threading
import multiprocessing
import copy
from datetime import timezone
import itertools
from core1 import * 

import datetime

from zoneinfo import ZoneInfo
try:
    from skyfield.api import load
except ImportError:
    import pip
    pip.main(['install', 'skyfield'])

def get(t: datetime.datetime):
    """_summary_

    Args:
        t (datetime.datetime): datetime object

    Returns:
        tuple: tuple containing:
            - numpy.array: 3D array containing the positions of the Earth and Moon relative to the Sun
            - numpy.array: 3D array containing the velocities of the Earth and Moon relative to the Sun
    """
    planets = load('de421.bsp')
    
    # Convert the provided datetime to a Skyfield time object
    ts = load.timescale()
   
    t = ts.from_datetime(t)
    
    # Get the celestial bodies
    earth = planets['earth barycenter']
    moon = planets['moon']
    sun = planets['sun']
    
    # Position data at the specified time
    earth_position = earth.at(t)
    moon_position = moon.at(t)
    sun_position = sun.at(t)
   
    # Calculate positions relative to the Sun
    pos_earth_rel_sun = earth_position.position.m - sun_position.position.m
    pos_moon_rel_sun = moon_position.position.m - sun_position.position.m

    # Calculate velocities relative to the Sun
    vel_earth_rel_sun = earth_position.velocity.m_per_s - sun_position.velocity.m_per_s
    vel_moon_rel_sun = moon_position.velocity.m_per_s - sun_position.velocity.m_per_s

    
    return np.array([[0,0,0],pos_earth_rel_sun, pos_moon_rel_sun]), np.array([[0,0,0],vel_earth_rel_sun, vel_moon_rel_sun])


# Correct eclipse data from 2010 to 2035
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


def does_line_intersect_before_circle(r, center, p1, direction1, p2, direction2):
    """
    Determines if a line intersects a circle before reaching the circle's boundary.

    Args:
        r : Radius of the circle.
        center : Coordinates of the center of the circle.
        p1 : Coordinates of a point on the line.
        direction1 : Direction vector of the line.
        p2 : Coordinates of a point on the line.
        direction2 : Direction vector of the line.
    """
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
    """
    Determines if a line intersects a half-circle.

    Args:
        r (float): Radius of the half-circle.
        center (numpy.array): Coordinates of the center of the half-circle.
        norm (numpy.array): Normal vector to the half-circle.
        point (numpy.array): Coordinates of a point on the line.
        direction (numpy.array): Direction vector of the line.

    Returns:
        _type_: _description_
    """
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
    """
    Determines the type of solar eclipse based on the positions of the Sun, Moon, and Earth.

    Args:
        sun_pos : Position vector of the Sun.
        moon_pos : Position vector of the Moon.
        earth_pos: Position vector of the Earth.
        sun_radius: Radius of the Sun.
        moon_radius : Radius of the Moon.
        earth_radius : Radius of the Earth.
        ep : _description_. Defaults to None.
      

    Returns:
        _type_: _description_
    """
    # Check if the Sun, Moon, and Earth are aligned
    align,gap=is_solar_eclipse_aligned(sun_pos, moon_pos, earth_pos,ep)
    if not align:
  
        return False, "none",None
    
    # Project the Sun, Moon, and Earth onto a plane
    sun_pos,moon_pos,earth_pos=project_to_plane(sun_pos,moon_pos,earth_pos)
    
    sun_to_moon = moon_pos - sun_pos
    sun_to_earth = earth_pos - sun_pos
    earth_to_sun = sun_pos - earth_pos
   
                                                                                                                                       


    sun_top = sun_pos +np.array( [0,1])* sun_radius
    sun_bot = sun_pos -np.array( [0,1])* sun_radius
    moon_top = moon_pos +np.array( [0,1])* moon_radius
    moon_bot = moon_pos -np.array( [0,1])* moon_radius
    
    # Calculate the positions of the penumbra and umbra
    penumbra_top_start = sun_top
    penumbra_top_direction = moon_bot - sun_top
    penumbra_bot_start = sun_bot
    penumbra_bot_direction = moon_top - sun_bot

    # Check if the penumbra is intersecting the Earth
    is_penumbra_top_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos, earth_to_sun,penumbra_top_start, penumbra_top_direction)
    is_penumbra_bot_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos,earth_to_sun, penumbra_bot_start, penumbra_bot_direction)

    umbra_top_start = sun_top
    umbra_top_direction = moon_top - sun_top
    umbra_bot_start = sun_bot
    umbra_bot_direction = moon_bot - sun_bot
    
    # Check if the umbra is intersecting the Earth
    is_umbra_top_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos,earth_to_sun,umbra_top_start, umbra_top_direction)
    is_umbra_bot_in_earth = does_line_intersect_half_circle(earth_radius, earth_pos,earth_to_sun,umbra_bot_start, umbra_bot_direction)
    
    is_partial_eclipse = is_penumbra_top_in_earth or is_penumbra_bot_in_earth 
   
    is_total_or_annular = is_umbra_top_in_earth or is_umbra_bot_in_earth
   
    if is_total_or_annular:
        
        # Check if the umbra is intersecting itself before the Earth
        line_intersect_before_half_circle=does_line_intersect_before_circle(earth_radius, earth_pos,umbra_top_start, umbra_top_direction, umbra_bot_start, umbra_bot_direction)
      
        return  True, "Annular" if line_intersect_before_half_circle else "Total", gap
    
   
    if is_partial_eclipse:
        return  True, "Partial", gap
    return False, "None",None
    
    
def find_closest_eclipse(eclipses, date,days_limit=float("inf")):
    """
    Find the closest eclipse to a given date.

    Args:
        eclipses: List of eclipse tuples.
        date: Date to compare against.
        days_limit: Maximum number of days to search for an eclipse.

    Returns:
        tuple: Closest eclipse tuple.
    """
    closest_eclipse = None
    closest_distance = float("inf")
    for eclipse in eclipses:
        distance = abs((eclipse[0] - date).days)
        if distance < closest_distance and distance<days_limit:
            closest_distance = distance
            closest_eclipse = eclipse
    return closest_eclipse

  
  
def find_all_eclipses_speed(x, v, m, dt, G,r, total_time, start_date):
    """ 
    Find all eclipses that occur during the simulation.
    """
    # print("total_time",total_time,dt)
    times = np.arange(0, total_time, dt)
    # print(len(times))
    # eclipse_types=set()
    current_eclipse_time_times=dict()
    eclipses=[]
 
    eclipse_start=False
    
    try:
        for current_time in times:
            
            #update positions
            verlet_update_all(x, v, m, dt, G, [True, True,True])
           
            x_sun, x_earth, x_moon = x
            result,type,gap=is_solar_eclipse(x_sun, x_moon, x_earth, r[0], r[2], r[1])
           
            if result:
                eclipse_start = True
              
                current_eclipse_time_times.update({gap:(type,current_time)})
            else:
                if eclipse_start:
                    eclipse_start = False
                     
                    # Find most aligned eclipse
                    info=min(current_eclipse_time_times.items(), key=lambda x: x[0])[1]
                    center_time=info[1]
                    eclipse_types=info[0]
                    center_time=start_date+datetime.timedelta(seconds=int(center_time))
                    eclipses.append((center_time,eclipse_types))
                  

    except Exception as e:
        print(e)
        pass
    return eclipses


def find_all_eclipses_speed_args(args):
    return find_all_eclipses_speed(*args)


def split_array(times, n):
    """
    Split an array into n equal parts.
    """
    k, m = divmod(len(times), n)
    return (times[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))



        
    

        
        
    
def print_eclipses(eclipses,correct_eclipses,start_date,total_time):
    """
    Plot the differences in days between the correct and found eclipses.
    """
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
    
    secs_per_worker=[total_time//splits for i in range(splits)]

    xs=[]
    vs=[]
    for i in range(splits):
        xs.append(get(start_date+datetime.timedelta(seconds=int( spited_times[i][0])))[0])
        vs.append(get(start_date+datetime.timedelta(seconds= int(spited_times[i][0])))[1])
        
    
  
   
    args=[(xs[i],vs[i],m,dt,G,r,secs_per_worker[i],date[i]) for i in range(splits)]

    pool = multiprocessing.Pool(processes=workers)
  
    results = pool.map(find_all_eclipses_speed_args, args)

    pool.close()
    pool.join()
    print("Eclipses Found")
    
    results=list(itertools.chain(*results))
   
    print_eclipses(results,correct_eclipses,start_date,total_time)
   

def main():
    main_speed()

if __name__ == "__main__":
    main()
    
