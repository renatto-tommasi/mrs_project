import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose

class StateValidityChecker:
    def __init__(self, robot_dim=(1, 1)):
        self.map = None
        self.map_dim = None
        self.robot_dim = robot_dim
        self.map_resolution = None

        rospy.Subscriber("/map", OccupancyGrid, self.saveMap)

    def saveMap(self, msg: OccupancyGrid):
        self.map = msg.data
        self.map_dim = (msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution

    def map_to_location(self, map_coordinates):
        location = Pose()
        location.position.x = (map_coordinates[1] - self.map_dim[1] / 2) * self.map_resolution + self.robot_dim[1] / 2
        location.position.y = (self.map_dim[0] / 2 - map_coordinates[0]) * self.map_resolution - self.robot_dim[0] / 2
        return location

    def location_to_map(self, location: Pose):
        map_coordinates = (int(self.map_dim[0] / 2 - (location.position.y + self.robot_dim[0] / 2) / self.map_resolution),
                           int((location.position.x - self.robot_dim[1] / 2) / self.map_resolution + self.map_dim[1] / 2))
        return map_coordinates

    def is_valid(self, map_coordinates):
        # Check if the coordinates are within the map boundaries
        if (map_coordinates[0] < 0 or map_coordinates[0] >= self.map_dim[0] or
            map_coordinates[1] < 0 or map_coordinates[1] >= self.map_dim[1]):
            return False

        # Check if the robot collides with any obstacles in the map
        for i in range(map_coordinates[0], map_coordinates[0] + self.robot_dim[0]):
            for j in range(map_coordinates[1], map_coordinates[1] + self.robot_dim[1]):
                if i >= 0 and i < self.map_dim[0] and j >= 0 and j < self.map_dim[1]:
                    if self.map[i * self.map_dim[1] + j] == 100:  # Assuming 100 represents an obstacle
                        return False
        return True
    