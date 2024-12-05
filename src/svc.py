import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt

class StateValidityChecker:

    def __init__(self, robot_dim=(1, 1)):
        self.map = None
        self.map_dim = None
        self.robot_dim = robot_dim
        self.map_resolution = None
        self.origin = None  # To store the map origin

        rospy.Subscriber("/map", OccupancyGrid, self.saveMap)

    def saveMap(self, msg: OccupancyGrid):
        """
        Callback function to save the occupancy grid map.

        Args:
          msg: The OccupancyGrid message received from the /map topic.
        """
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_dim = (msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)  # Save origin
        

    

    def is_state_valid(self, state):
        """
        Checks if a given state (x, y) is valid within the map and considering 
        robot dimensions.

        Args:
          state: A tuple (x, y) representing the state to check.

        Returns:
          True if the state is valid, False otherwise.
        """
        if self.map is None:
            return False  # No map received yet

        x, y = state
        # Convert map coordinates to grid indices
        row, col = self.map_to_grid(x, y)

        # Check if the cell is free
        if self.get_cell_value(row, col) == 0:
            # Check if the robot footprint is free
            return self.check_robot_footprint(row, col)
        return False

    def map_to_grid(self, x, y):
        """
        Converts map coordinates (x, y) to grid indices (row, col) with 
        origin at the top-left.

        Args:
          x: The x-coordinate on the map.
          y: The y-coordinate on the map.

        Returns:
          A tuple (row, col) representing the grid indices.
        """
        # Adjust for origin and invert y to match image coordinates
        col = int(np.floor((x - self.origin[0]) / self.map_resolution))
        row = int(np.floor((-self.origin[1]+y) / self.map_resolution)) 
        return (row, col)
    
    def grid_to_map(self, row, col):
      """
      Converts grid indices (row, col) to map coordinates (x, y) in the world frame.

      Args:
        row: The row index in the grid.
        col: The column index in the grid.

      Returns:
        A tuple (x, y) representing the corresponding map coordinates.
      """
      # Map dimensions (center coordinates in grid units)
      center_row = self.map_dim[0] / 2
      center_col = self.map_dim[1] / 2

      # Calculate real-world coordinates
      x = self.origin[0] + col * self.map_resolution
      y = self.origin[1] + row * self.map_resolution
      return (x, y)

    def get_cell_value(self, row, col):
        """
        Gets the value of a cell in the occupancy grid.

        Args:
          row: The row index.
          col: The column index.

        Returns:
          The value of the cell (-1 for unknown, 0 for free, 100 for occupied).
        """
        if 0 <= row < self.map_dim[0] and 0 <= col < self.map_dim[1]:
            return self.map[row * self.map_dim[1] + col]
        else:
            return 100  # Treat out-of-bounds cells as occupied

    def check_robot_footprint(self, row, col):
        """
        Checks if the robot's footprint is free of obstacles at the given cell.

        Args:
          row: The row index of the cell.
          col: The column index of the cell.

        Returns:
          True if the robot's footprint is free, False otherwise.
        """
        robot_rows = int(np.ceil(self.robot_dim[0] / self.map_resolution))
        robot_cols = int(np.ceil(self.robot_dim[1] / self.map_resolution))

        for i in range(row - robot_rows // 2, row + robot_rows // 2 + 1):
            for j in range(col - robot_cols // 2, col + robot_cols // 2 + 1):
                if self.get_cell_value(i, j) != 0:
                    return False
        return True