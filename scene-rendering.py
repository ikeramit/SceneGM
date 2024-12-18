
import numpy as np

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

# Define wall and camera parameters
focal_length = 35
image_size = (800, 600)  # in pixels

# Generate 3D points on the wall
'''wall_width = 10
wall_height = 5
wall_distance = 30
x = np.linspace(-wall_width / 2, wall_width / 2, 2)
y = np.linspace(-wall_height / 2, wall_height / 2, 2)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, wall_distance)
wall_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T'''

wall_points = np.array([[-5.0, -2.5, 30.0], [ 5.0, -2.5, 30.0], [-5.0, 2.5, 30.0], [5.0, 2.5, 30.0]])

# Define camera position and orientation
camera_position = np.array([0.0, 0.0, 0.0])  # Camera at the origin
# Camera looking towards positive Z with no rotation
camera_orientation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  [0.0, 0.0, 1.0]])

# Project wall points to the image plane
image_points = project_points(wall_points, camera_position, camera_orientation, focal_length, image_size)
print(image_points)

