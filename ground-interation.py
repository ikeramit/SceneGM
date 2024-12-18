
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def line_of_sight_ground_intersection(camera_position, direction_vector, dem, dem_x, dem_y):
    """
    Computes the intersection of a line of sight with the ground (DEM).

    Parameters:
    - camera_position: np.array of shape (3,) [x, y, z] for the camera's position.
    - direction_vector: np.array of shape (3,) [dx, dy, dz] for the line of sight's unit direction vector.
    - dem: 2D array of ground elevation values.
    - dem_x: 1D array of x-coordinates of the DEM grid.
    - dem_y: 1D array of y-coordinates of the DEM grid.

    Returns:
    - intersection_point: np.array [x, y, z] of the intersection point, or None if no intersection is found.
    """
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Create an interpolator for the DEM
    dem_interpolator = RegularGridInterpolator((dem_x, dem_y), dem.T, bounds_error=False, fill_value=None)

    # Step size for sampling along the line of sight
    step_size = 1.0  # Adjust for resolution (meters per step)
    max_steps = 10000  # Max iterations to prevent infinite loop

    # Initialize parameters
    t = 0.0  # Parameter along the line of sight
    for _ in range(max_steps):
        # Compute the current point along the line
        current_point = camera_position + t * direction_vector
        x, y, z = current_point

        # Interpolate ground height at the current x, y
        ground_height = dem_interpolator((x, y))

        if ground_height is None:
            # The line of sight is outside the DEM bounds
            return None

        # Check if the line is below the ground
        if z <= ground_height:
            return np.array([x, y, ground_height])

        # Increment t to move along the line
        t += step_size

    # If no intersection is found within the max steps
    return None

# Example usage
if __name__ == "__main__":
    # Define camera position and line of sight direction
    camera_position = np.array([0.0, 0.0, 1000.0])  # Camera at 1000m altitude
    direction_vector = np.array([0.0, 1.0, -1.0])  # Line of sight pointing downward at an angle

    # Define a simple DEM
    dem_x = np.linspace(-1000, 1000, 100)  # X-coordinates
    dem_y = np.linspace(-1000, 1000, 100)  # Y-coordinates
    dem = np.zeros((100, 100))  # Flat ground at 0m elevation

    # Compute the intersection
    intersection = line_of_sight_ground_intersection(camera_position, direction_vector, dem, dem_x, dem_y)

    if intersection is not None:
        print(f"Intersection point: {intersection}")
    else:
        print("No intersection found.")

'''#GENERATING A SIMPLE DEM
import numpy as np

# Define the grid extent and resolution
x_min, x_max = 0, 10  # x-coordinates range
y_min, y_max = 0, 10  # y-coordinates range
resolution = 1  # 1 meter per grid cell

# Generate grid coordinates
dem_x = np.arange(x_min, x_max + resolution, resolution)
dem_y = np.arange(y_min, y_max + resolution, resolution)

# Create a simple flat terrain DEM with some variation
dem = np.zeros((len(dem_y), len(dem_x)))  # Flat ground
dem += np.random.uniform(0, 2, size=dem.shape)  # Add small elevation variation

import matplotlib.pyplot as plt

plt.imshow(dem, extent=(dem_x[0], dem_x[-1], dem_y[0], dem_y[-1]), origin='lower', cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('Digital Elevation Model')
plt.show()'''
