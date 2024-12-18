
import numpy as np
import matplotlib.pyplot as plt

def calculate_shadow_from_rectangle(wall_points, sun_altitude, sun_azimuth):
    """
    Computes the shadow of a rectangular wall on the ground.

    Parameters:
    - wall_points: List of 4 tuples (x, y, z) defining the corners of the wall in 3D.
    - sun_altitude: Solar altitude angle (degrees, 0 = horizon, 90 = zenith).
    - sun_azimuth: Solar azimuth angle (degrees, 0 = North, clockwise).

    Returns:
    - shadow_polygon: List of 4 points (x, y) defining the shadow polygon on the ground.
    """
    # Convert sun angles to radians
    sun_altitude_rad = np.radians(sun_altitude)
    sun_azimuth_rad = np.radians(sun_azimuth)

    # Sunlight direction vector
    sunlight_dir = np.array([
        -np.sin(sun_azimuth_rad) * np.cos(sun_altitude_rad),  # x-component
        -np.cos(sun_azimuth_rad) * np.cos(sun_altitude_rad),  # y-component
        -np.sin(sun_altitude_rad)                             # z-component
    ])

    shadow_points = []
    for point in wall_points:
        # Find the intersection of the sunlight direction with the ground (z=0)
        t = -point[2] / sunlight_dir[2]  # Solve for t when z = 0
        shadow_point = np.array(point[:2]) + t * sunlight_dir[:2]  # Intersection (x, y)
        shadow_points.append(tuple(shadow_point))

    return shadow_points

def plot_wall_and_shadow(wall_points, shadow_polygon):
    """
    Plots the 3D wall and its shadow on the ground.
    """
    wall_x, wall_y, wall_z = zip(*wall_points)
    shadow_x, shadow_y = zip(*shadow_polygon)

    # Plot wall
    plt.plot(wall_x + (wall_x[0],), wall_y + (wall_y[0],), 'b-', label="Wall Base")

    # Plot shadow
    plt.fill(shadow_x + (shadow_x[0],), shadow_y + (shadow_y[0],), color='gray', alpha=0.5, label="Shadow")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Wall and Shadow")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define the wall as a rectangle with 4 corner points (x, y, z)
    wall_points = [
        (0, 0, 0),  # Bottom-left
        (1, 0, 0),  # Bottom-right
        (1, 0, 5),  # Top-right
        (0, 0, 5)   # Top-left
    ]

    # Sun parameters
    sun_altitude = 30  # Sun's angle above the horizon
    sun_azimuth = 135  # Sun's angle clockwise from North

    # Calculate shadow
    shadow_polygon = calculate_shadow_from_rectangle(wall_points, sun_altitude, sun_azimuth)

    # Plot results
    plot_wall_and_shadow(wall_points, shadow_polygon)
	