#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np

from geometry_msgs.msg import Pose, Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from dataclasses import dataclass
from geometry_msgs.msg import Point
import math
from svc import StateValidityChecker

@dataclass
class BoidState:
    pos: np.array
    vel: np.array
    dist: float

# Predefined colors for velocity types
BEHAVIOR_COLORS = [
    (1.0, 0.0, 0.0, 1.0),  # Red Cohesion
    (0.0, 1.0, 0.0, 1.0),  # Green Separation
    (0.0, 0.0, 1.0, 1.0),  # Blue allignment
    (1.0, 1.0, 0.0, 1.0),  # Yellow migration
    (1.0, 0.0, 1.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0, 1.0),  # Cyan
]


class BoidController:
    def __init__(self, id, num_of_robots, k_all=0.2, k_sep=0.3, k_coh=0.6, k_mig=0.8, k_obs=2):
        # Tuning Parameters
        self.k_sep = k_sep
        self.k_all = k_all
        self.k_coh = k_coh
        self.k_mig = k_mig
        self.k_obs = k_obs

        self.id = id
        self.neighbors = {}
        self.p_wf = np.array([0.0,0.0,0.0]).reshape(3,1)       # Boid Position in World Frame x , y , theta
        self.orientation = 0.0
        
        self.vel = np.array([0.0,0.0]).reshape(2,1)
        self.state_checker = StateValidityChecker()

        self.migration_target = None

        self.max_speed = 1

        # Publishers
        self.vel_pub = rospy.Publisher("/robot_{}/cmd_vel".format(self.id), Twist, queue_size=10)
        self.marker_pub = rospy.Publisher("/robot_{}/velocity_marker".format(self.id), MarkerArray, queue_size=10)


        # Subscriber
        rospy.Subscriber("/robot_{}/odom".format(self.id), Odometry, self.get_odom)
        for i in range(num_of_robots):
            if i != id:
                rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.get_neighbors, callback_args=i)

        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.migratory_urge)

        self.dt = 0.1

    def get_odom(self, odom:Odometry):
        '''
        Saves its own Odometry 
        '''
        p_x_wf = odom.pose.pose.position.x
        p_y_wf = odom.pose.pose.position.y
        vl_x = odom.twist.twist.linear.x
        vl_y = odom.twist.twist.linear.y

        self.vel = np.array([vl_x, vl_y]).reshape(2,1)

        self.orientation = self.getOrientation(self.vel)
        self.p_wf = np.array([p_x_wf, p_y_wf, self.orientation]).reshape(3,1)
        self.vel = np.array([vl_x, vl_y]).reshape(2,1)

        
        # rospy.loginfo(f"Boid {self.id} Position: {self.p_wf}")
        
        self.reynolds()
        # else:
            # self.vel = np.array([0.0,0.0]).reshape(2,1)
            # self.publishVel()

    def migratory_urge(self, goal:PoseStamped):
        x_goal_wf = goal.pose.position.x
        y_goal_wf = goal.pose.position.y
        
        self.migration_target = np.array([x_goal_wf,y_goal_wf]).reshape(2,1)

        # rospy.loginfo(f"Received Goal at {self.migration_target}")

    def get_neighbors(self, n_odom: Odometry, n_id):
        """
        Identifies neighbors of the robot in its own reference frame.
        """
        n_pos_rf = np.array([0.0,0.0,0.0]).reshape(3,1)
        n_x_wf = n_odom.pose.pose.position.x
        n_y_wf = n_odom.pose.pose.position.y
        vel = self.get_neighbors_vel(n_odom)


        n_pos_wf = np.array([n_x_wf, n_y_wf]).reshape(2, 1)

        n_pos_rf[:2] = n_pos_wf - self.p_wf[:2]
        n_pos_rf[2] = np.arctan2(n_pos_rf[1], n_pos_rf[0]) - self.orientation
        
        n_dist = np.linalg.norm(n_pos_rf[:2])

        if self.isVisible(n_dist, n_pos_rf[2]):
            # rospy.loginfo(f"Neighbor {n_id} is visible from Boid {self.id} at {n_pos_rf[2]}")
            n = BoidState(pos=n_pos_rf, vel=vel, dist=n_dist)
            self.neighbors[n_id] = n

            
        else:
            # rospy.loginfo(f"Neighbor {n_id} is NOT visible from Boid {self.id} at {n_pos_rf[2]}")
            if n_id in self.neighbors:
                del self.neighbors[n_id]
                # rospy.loginfo(f"Neighbor {n_id} removed")
    
    def isVisible(self, n_dist, angle, max_distance = 2.0, field_of_view = np.pi):
        if n_dist > max_distance:  # Neighbor is too far
            rospy.loginfo("Too far away!!")
            return False
        return -field_of_view <= angle <= field_of_view
    def getOrientation(self, vel):
        if np.linalg.norm(vel) > 0:
            self.orientation =  np.arctan2(self.vel[1], self.vel[0])[0]
        return self.orientation
    def get_neighbors_vel(self, n_odom:Odometry):
        '''
        This function simulates a way of getting the velocities of the neighboring robots.
        It first identifies the neighbors and then stores their velocity
        '''
        return np.array([n_odom.twist.twist.linear.x, n_odom.twist.twist.linear.y]).reshape(2,1)
        

    def reynolds(self):
        allignment_acc = self.getAllignmentAcc()  
        cohesion_acc = self.getCohesionAcc()
        separation_acc = self.getSeparationAcc()
        migration_acc = self.getMigrationAcc()
        obstacle_acc = self.getObstacleAvoidance()

        vel = self.calculateVel(allignment_acc, cohesion_acc, separation_acc, migration_acc, obstacle_acc)

        speed = np.linalg.norm(vel)
        if speed > self.max_speed:
            vel = vel/speed * self.max_speed


        self.publishVel(vel)



    def getAllignmentAcc(self):
        '''
        Matches the velocity of the neighbors
        '''
        allignment_vel = np.zeros((2,1))

        vel_all = np.zeros((2,1))
        
        if len(self.neighbors) > 0:
            for _, agent in self.neighbors.items():
                vel_all += agent.vel
            # Normalize by total weight if it's greater than zero
            allignment_vel = vel_all / len(self.neighbors)

        allignment_acc = allignment_vel - self.vel


        return allignment_acc   
    
    def getCohesionAcc(self):
        coh_acc = np.zeros((2,1))
        center_of_mass = np.zeros((2, 1))  # Initialize center of mass to zero
        num_neighbors = len(self.neighbors)

        if num_neighbors > 0:
            # Calculate the center of mass
            for _, agent in self.neighbors.items():
                center_of_mass += agent.pos[:2]
            center_of_mass /= num_neighbors + 1  # Average position of neighbors

        coh_acc = center_of_mass 
        return coh_acc
    
    def getSeparationAcc(self):
        sep_acc = np.zeros((2,1))
        num_neighbors = len(self.neighbors)

        if num_neighbors > 0:
            for _, agent in self.neighbors.items():
                direction_vector = -agent.pos[:2] / agent.dist

                sep_acc += (1/agent.dist**2) * direction_vector
            sep_acc /= len(self.neighbors)
        

        return sep_acc
    
    def getMigrationAcc(self):

        if self.migration_target is None:
            return np.zeros((2,1))
        
        vector = self.migration_target - self.p_wf[:2]
        mig_acc = vector / np.linalg.norm(vector)

        

        return mig_acc * np.linalg.norm(vector)**2
    
    def getObstacleAvoidance(self):
        obs_acc = np.zeros((2, 1))             # Initialize acceleration vector
        obstacle_distance = 0.4                  # The maximum distance to check for obstacles
        resolution = self.state_checker.map_resolution
        map_data = self.state_checker.map
        map_dim = self.state_checker.map_dim
        map_obstacle_threshold = 5                  # Threshold for considering an obstacle
        grid_x, grid_y = self.state_checker.map_to_grid(self.p_wf[0], self.p_wf[0])
        radius = int(obstacle_distance / resolution) # Radius
        obstacle_count = 0
        # Iterate through the grid
        for iy in range(grid_y - radius, grid_y + radius + 1):
            for ix in range(grid_x - radius, grid_x + radius + 1):
                if 0 <= ix < map_dim[1] and 0 <= iy < map_dim[0]:                      # Check if the grid point is within the map bounds
                    if map_data[iy, ix] >= map_obstacle_threshold:  
                        direction_x = grid_x - ix                             
                        direction_y = grid_y - iy                           
                        distance = math.sqrt(direction_x**2 + direction_y**2)
                        if distance < 1e-3:
                            distance =  1e-3 # Compute direction from robot to the obstacle
                        obs_acc[0] += (direction_x / distance) / distance
                        obs_acc[1] += (direction_y / distance) / distance
                        obstacle_count += 1
        if obstacle_count > 0:
            obs_acc /= obstacle_count

        return obs_acc

    
    def calculateVel(self, allignment_acc, cohesion_acc, separation_acc, migration_acc, obstacle_acc):
        
        force_acc = self.k_coh * cohesion_acc + self.k_sep * separation_acc + self.k_all * allignment_acc + self.k_mig * migration_acc + self.k_obs * obstacle_acc

        agent_vel = force_acc 

        velocities = [self.k_coh * cohesion_acc,
                      self.k_sep * separation_acc,
                      self.k_all * allignment_acc,
                      self.k_mig * migration_acc,
                      self.k_obs * obstacle_acc,
                      agent_vel 

                      ]

        self.publish_velocity_arrows(self.p_wf , velocities)


        return agent_vel
    
    def publishVel(self, vel):

        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]

        self.vel_pub.publish(vel_msg)

    def publish_velocity_arrows(self, origin, velocities):
        """
        Publishes an RViz marker array representing velocities as arrows.

        Args:
            origin (tuple): The (x, y, z) coordinates of the arrows' base.
            velocities (list of numpy arrays): A list of velocity vectors (np.array([vx, vy])).
            velocity_types (list of str): The corresponding behavior types for each velocity.
            marker_pub (rospy.Publisher): The ROS publisher for the marker array.
        """
        marker_array = MarkerArray()

        for i, velocity in enumerate(velocities):
            # Create a unique marker for each velocity
            marker = Marker()
            marker.header.frame_id = f"robot_{self.id}/base_link"  # Replace with your frame ID
            marker.header.stamp = rospy.Time.now()
            marker.ns = "velocity_arrows"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Define the start and end points of the arrow
            start_point = Point()
            start_point.x = 0.0
            start_point.y = 0.0
            start_point.z = 0.0
            end_point = Point(velocity[0], velocity[1], 0)
            marker.points.append(start_point)
            marker.points.append(end_point)

            # Set arrow properties
            marker.scale.x = 0.05  # Shaft diameter
            marker.scale.y = 0.1  # Head diameter
            marker.scale.z = 0.1  # Head length

            # Assign a predefined color based on the velocity type
            
            r, g, b, a = BEHAVIOR_COLORS[i]

            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = a

            # Add the marker to the array
            marker_array.markers.append(marker)

        # Publish the marker array
        self.marker_pub.publish(marker_array)



if __name__ == '__main__':
    rospy.init_node('swarm_controller_node')

    num_of_robots = rospy.get_param("/num_of_robots")
    
    swarm = [BoidController(i, num_of_robots) for i in range(num_of_robots)]

    rospy.spin()
