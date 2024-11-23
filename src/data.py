from geometry_msgs.msg import Pose, Twist
import tf.transformations
from dataclasses import dataclass

import numpy as np


def conv_quaternion_theta(q:Pose):
            euler = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            return euler[2]  # Extract the yaw angle 

@dataclass
class BoidState:
    pos: np.array
    vel: np.array
    dist: float


def rotate_matrix(matrix, theta):
  """
  Rotates a 2D binary matrix by an angle theta (in radians) 
  around the origin (0, 0).

  Args:
    matrix: A NumPy array representing the binary matrix.
    theta: The angle of rotation in radians.

  Returns:
    A new NumPy array representing the rotated matrix.
  """

  rows, cols = matrix.shape
  rotated_matrix = np.zeros((rows, cols), dtype=int)  # Initialize a new matrix 
  cos_theta = np.cos(theta)
  sin_theta = np.sin(theta)

  for x in range(rows):
    for y in range(cols):
      # Apply rotation transformation
      x_new = int(np.round(x * cos_theta - y * sin_theta))
      y_new = int(np.round(x * sin_theta + y * cos_theta))

      # Check boundaries and assign value
      if 0 <= x_new < rows and 0 <= y_new < cols:
        rotated_matrix[x_new, y_new] = matrix[x, y]

  return rotated_matrix

