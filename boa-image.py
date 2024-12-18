
import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from matplotlib.path import Path

#*****************************
#********* GEOMETRY **********
#*****************************

def rotation_matrix_3d(theta_x, theta_y, theta_z):
  # Convert angles to radians
  theta_x = np.radians(theta_x)
  theta_y = np.radians(theta_y)
  theta_z = np.radians(theta_z)
  # Rotation matrix around the X-axis
  R_x = np.array([
      [1, 0, 0],
      [0, np.cos(theta_x), -np.sin(theta_x)],
      [0, np.sin(theta_x), np.cos(theta_x)]
  ])
  # Rotation matrix around the Y-axis
  R_y = np.array([
      [np.cos(theta_y), 0, np.sin(theta_y)],
      [0, 1, 0],
      [-np.sin(theta_y), 0, np.cos(theta_y)]
  ])
  # Rotation matrix around the Z-axis
  R_z = np.array([
      [np.cos(theta_z), -np.sin(theta_z), 0],
      [np.sin(theta_z), np.cos(theta_z), 0],
      [0, 0, 1]
  ])
  # Combined rotation matrix (Rz * Ry * Rx)
  R = R_z @ R_y @ R_x
  return R
def project_points(points, camera_position, camera_orientation, focal_length, image_size):
  """
  Projects 3D points onto a 2D image plane using a pinhole camera model.
  Parameters:
      points (ndarray): Nx3 array of 3D points on the wall.
      camera_position (ndarray): 1x3 array for the camera position in 3D space.
      camera_orientation (ndarray): 3x3 rotation matrix for the camera orientation.
      focal_length (float): Distance from the camera center to the image plane.
      image_size (tuple): Width and height of the image plane in pixels.
  Returns:
      projected_points (ndarray): Nx2 array of 2D points on the image plane.
  """
  # Translate points relative to the camera position
  translated_points = points - camera_position
  # Rotate points to align with the camera coordinate system
  rotated_points = translated_points @ camera_orientation.T
  # Project points onto the image plane
  projected_points = focal_length * rotated_points[:, :2] / rotated_points[:, 2:3]
  # Convert to image coordinates
  image_center = np.array(image_size) / 2
  projected_points = projected_points + image_center
  return projected_points
def mask_array_with_polygon(array, polygon_points, value):
  """
  Masks a 2D numpy array by setting cells within a polygon to 1.
  Parameters:
  - array: 2D numpy array (e.g., 100x100) to be masked, initially filled with np.nan.
  - polygon_points: A 2D numpy array of shape (N, 2) representing the vertices of the polygon.
  Returns:
  - A 2D numpy array with cells inside the polygon set to 1.
  """
  rows, cols = array.shape
  # Create a grid of x, y coordinates corresponding to array indices
  x, y = np.meshgrid(np.arange(cols), np.arange(rows))
  points = np.vstack((x.ravel(), y.ravel())).T
  # Create a Path object for the polygon
  polygon_path = Path(polygon_points)
  # Find which points are inside the polygon
  mask = polygon_path.contains_points(points)
  # Reshape the mask to the array's shape
  mask = mask.reshape(rows, cols)
  # Set values inside the polygon to 1
  array[mask] = value
  return array
def add_terrain(array, x, y, width, length, temperature):
  array.append({"value": temperature, "wall_points": np.array([[x, y, 0.0], [x + width, y, 0.0], [x + width, y + length, 0.0], [x, y + length, 0.0]])})
  return array
def add_building_right(array, x, y, width, length, height, temperature):
  array.append({"value": temperature - 2.0, "wall_points": np.array([[x, y, 0.0], [ x + width, y, 0.0], [x + width, y, height], [x, y, height]])})
  array.append({"value": temperature + 2.0, "wall_points": np.array([[x, y, 0.0], [ x, y + length, 0.0], [x, y + length, height], [x, y, height]])})
  array.append({"value": temperature, "wall_points": np.array([[x, y, height], [x + width, y, height], [x + width, y + length, height], [x, y + length, height]])})
  return array
def add_building_left(array, x, y, width, length, height, temperature):
  array.append({"value": temperature + 2.0, "wall_points": np.array([[x + width, y, 0.0], [ x + width, y + length, 0.0], [x + width, y + length, height], [x + width, y, height]])})
  array.append({"value": temperature - 2.0, "wall_points": np.array([[x, y, 0.0], [ x + width, y, 0.0], [x + width, y, height], [x, y, height]])})
  array.append({"value": temperature, "wall_points": np.array([[x, y, height], [x + width, y, height], [x + width, y + length, height], [x, y + length, height]])})
  return array

#*********************************************
#********** RADIATIVE HEAT TRANSFER **********
#*********************************************

def planck_law(wavelength, temperature):
    """
    Computes spectral radiance using Planck's Law.
    Parameters:
        wavelength (ndarray): Wavelengths in meters.
        temperature (float): Temperature in Kelvin.
    Returns:
        ndarray: Spectral radiance (W/m^2/sr/m).
    """
    term1 = (2 * h * c**2) / (wavelength**5)
    term2 = 1 / (np.exp((h * c) / (wavelength * k_B * temperature)) - 1)
    return term1 * term2
def total_radiance(temperature, emissivity=1.0, absorption_coefficient=0.01, background_temp=300, begin_wl=8e-6, end_wl=14e-6):
    """
    Computes total radiance measured by the camera.
    Parameters:
        temperature (float): Object temperature (K).
        distance (float): Distance to the object (m).
        emissivity (float): Emissivity of the object (0 to 1).
        absorption_coefficient (float): Atmospheric absorption coefficient.
        background_temp (float): Background temperature (K).
    Returns:
        float: Total radiance measured by the camera (W/m^2/sr).
    """
    # Wavelength range for thermal cameras (8 µm to 14 µm)
    wavelengths = np.linspace(begin_wl, end_wl, 1000)  # in meters
    # Radiance from the object
    radiance_object = np.trapz(planck_law(wavelengths, temperature), wavelengths)
    # Radiance from the background
    radiance_background = np.trapz(planck_law(wavelengths, background_temp), wavelengths)
    # Total radiance measured by the camera
    radiance_measured = transmissivity * (emissivity * radiance_object / distance**2) + (1 - transmissivity) * radiance_background
    return radiance_measured

# Constants
h = 6.626e-34  # Planck's constant (J·s)
c = 3e8        # Speed of light (m/s)
k_B = 1.38e-23 # Boltzmann constant (J/K)

# Example: Compute radiance for a point at a distance
object_temp = 350       # Temperature of the object (K)
transmissivity = 0.7           # Atmospheric transmissivity
emissivity = 0.95       # Emissivity of the object
absorption_coefficient = 0.01  # Atmospheric absorption coefficient
background_temp = 290   # Background temperature (K)

# Define wall and camera parameters
focal_length = 50
image_size = (200, 200)  # in pixels

objects = []
objects = add_terrain(objects, -90.0, 0.0, 180.0, 120.0, 20.0)

objects = add_building_right(objects, 50.0, 30.0, 20.0, 60.0, 30.0, 25.0)
objects = add_building_right(objects, 0.0, 10.0, 20.0, 30.0, 30.0, 25.0)
objects = add_building_left(objects, -70.0, 50.0, 50.0, 30.0, 30.0, 25.0)

#objects = add_building_right(objects, 20.0, 40.0, 50.0, 30.0, 30.0, 25.0)
#objects = add_building_left(objects, -20.0, 80.0, 20.0, 30.0, 30.0, 25.0)
#objects = add_building_left(objects, -70.0, 30.0, 20.0, 60.0, 30.0, 25.0)

# Define camera position and orientation
camera_position = np.array([-10.0, -15.0, 100.0])  # Camera at the origin
# Camera looking towards positive Z with no rotation
theta_x = -50.0
theta_y = 0.0
theta_z = 10.0
camera_orientation = np.array(rotation_matrix_3d(theta_x, theta_y, theta_z))

# Create a 100x100 array filled with np.nan
array = np.full(image_size, total_radiance(273.16 + 15, emissivity, absorption_coefficient, background_temp, 8e-6, 14e-6))

for obj in objects:
# Project wall points to the image plane
  image_points = project_points(obj['wall_points'], camera_position, camera_orientation, focal_length, image_size)
  #print(image_points)

  # Mask the array
  array = mask_array_with_polygon(array, image_points, total_radiance(273.16 + obj['value'], emissivity, absorption_coefficient, background_temp, 8e-6, 14e-6)).astype(np.float32)
  #print(masked_array)

array = np.flip(array, 1)

plt.imshow(array, interpolation='nearest')
plt.show()

# Specify the band name in the metadata
band_name = "Band 1: Float Data Example"

# Save the array with metadata
tiff.imwrite(
    'scout/output_with_band_name.tiff',
    array,
    description=band_name  # Add band name as a description
)
