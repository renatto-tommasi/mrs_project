#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, PoseStamped
import numpy as np
import tf.transformations

from svc import StateValidityChecker
from data import conv_quaternion_theta, BoidState, rotate_matrix

class BoidController:
    def __init__(self, id, num_of_robots, k_all=0.7, k_sep=0.005, k_coh=0.2, k_mig=0.1, k_obs=0.0001):
        # Tuning Parameters
        self.k_sep = k_sep
        self.k_all = k_all
        self.k_coh = k_coh
        self.k_mig = k_mig
        self.k_obs = k_obs


        # Velocities
        self.separation_vel = np.zeros((2,1))
        self.allignment_vel = np.zeros((2,1))
        self.cohesion_vel = np.zeros((2,1))
        self.migration_vel = np.zeros((2, 1))
        self.repulsive_vel = np.zeros((2, 1)) 
        
        self.max_velocity = 3


        self.pose_wf = Pose()
        self.vel = Twist()
        self.id = id
        self.neighbors = {}
        self.p_wf = np.array([0,0,0])
        self.svc = StateValidityChecker()

        
        self.migration_target = None

        # Publishers
        self.vel_pub = rospy.Publisher("/robot_{}/cmd_vel".format(self.id), Twist, queue_size=10)

        # Subscriber
        rospy.Subscriber("/robot_{}/odom".format(self.id), Odometry, self.get_odom)
        for i in range(num_of_robots):
            if i != id:
                rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.get_neighbors, callback_args=i)

        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.migratory_urge)
        # Timers
        rospy.Timer(rospy.Duration(5), self.print_neighbors_vel) 

    def migratory_urge(self, goal:PoseStamped):
        x_goal_wf = goal.pose.position.x
        y_goal_wf = goal.pose.position.y

        self.migration_target = np.array([x_goal_wf,y_goal_wf]).reshape(2,1)


    def isVisible(self, pos_rf:np.array, n_dist:float):
        return n_dist < 1 and -3/4*np.pi <= pos_rf[2] < 3/4*np.pi
    
    def get_odom(self, odom:Odometry):
        '''

        Saves its own Odometry 

        '''
        p_x_wf = odom.pose.pose.position.x
        p_y_wf = odom.pose.pose.position.y
        p_theta_wf = conv_quaternion_theta(odom.pose.pose.orientation)
        self.p_wf = np.array([p_x_wf, p_y_wf, p_theta_wf]).reshape(3,1)

        self.pose = odom.pose.pose
        self.vel = odom.twist.twist

        self.reynolds()

    def get_neighbors(self, n_odom:Odometry, n_id):
        # Transform coordinates from World to Robot Frame
        n_x_wf = n_odom.pose.pose.position.x
        n_y_wf = n_odom.pose.pose.position.y
        n_theta_wf = conv_quaternion_theta(n_odom.pose.pose.orientation)
        n_pos_wf = np.array([n_x_wf, n_y_wf, n_theta_wf]).reshape(3,1)
        n_pos_rf = n_pos_wf - self.p_wf
        # Calculate distance to boid
        n_dist = np.linalg.norm(n_pos_rf[:2])

        # Is considered as a neighbor if its within the visibility range
        if self.isVisible(n_pos_rf, n_dist):
            vel = self.get_neighbors_vel(n_odom, n_id)
            n = BoidState(pos=n_pos_rf, vel=vel, dist=n_dist)
            self.neighbors[n_id] = n
        else:
            if n_id in self.neighbors:
                del self.neighbors[n_id]
                
    def get_neighbors_vel(self, n_odom:Odometry, n_id):
        '''
        This function simulates a way of getting the velocities of the neighboring robots.
        It first identifies the neighbors and then stores their velocity
        '''

        dtheta = - conv_quaternion_theta(self.pose.orientation) + conv_quaternion_theta(n_odom.pose.pose.orientation)
        R = np.array([[np.cos(dtheta), -np.sin(dtheta)], 
                  [np.sin(dtheta), np.cos(dtheta)]])

        # Store the neighbor's velocity and distance
        vel = np.array([n_odom.twist.twist.linear.x, n_odom.twist.twist.linear.y])
        transformed_vel = R @ vel.T
        return transformed_vel
    
    def print_neighbors_vel(self, _):
        """Prints the neighbor velocity dictionary to the terminal."""
        rospy.loginfo("Robot ID: %s, Neighbors: %s", self.id, self.neighbors)
    
    def publish_vel(self):
        agent_vel = self.allignment_vel + self.separation_vel + self.cohesion_vel + self.migration_vel + self.repulsive_vel

        vel_msg = Twist()
        vel_msg.linear.x = agent_vel[0][0]/5
        vel_msg.linear.y = agent_vel[1][0]/5

        self.vel_pub.publish(vel_msg)



    def reynolds(self):
        
        self.allignment()
        self.separation()
        self.cohesion()
        self.migration() 
        # TODO: Uncomment to test your osbtacle avoidance program
        # self.obstacle_avoidance()
        self.publish_vel()


    def separation(self):
        self.separation_vel = np.zeros((2,1))

        for _, agent in self.neighbors.items():
            if 0 < agent.dist < 0.3:  # Only consider neighbors within the threshold
                # Calculate the unit direction vector (opposite to position)
                direction_vector = -agent.pos[:2] / agent.dist  

                # Calculate the velocity magnitude (quadratically proportional to inverse distance)
                velocity_magnitude = self.max_velocity / (agent.dist ** 2)  

                # Calculate the velocity vector
                self.separation_vel += velocity_magnitude * direction_vector

        self.separation_vel *= self.k_sep
        

    def allignment(self):
        '''

        Matches the velocity of the neighbors
        
        '''
        # Transform the neighbors velocity from the neighborframe to the body frame
        total_weight = 0.0
        weighted_linear_x = 0.0
        weighted_linear_y = 0.0
        self.allignment_vel = np.zeros((2,1))

        for _, agent in self.neighbors.items():
            if agent.dist > 0:  # Avoid division by zero
                weight = 1.0 / agent.dist
                total_weight += weight
                weighted_linear_x += agent.vel[0] * weight
                weighted_linear_y += agent.vel[1] * weight

        # Normalize by total weight if it's greater than zero
        if total_weight > 0:
            self.allignment_vel = np.array([weighted_linear_x / total_weight, weighted_linear_y / total_weight])

        self.allignment_vel = self.k_all * self.allignment_vel   

    def cohesion(self):
        """
        Steers the robot towards the center of mass of its neighbors.
        """
        self.cohesion_vel = np.zeros((2, 1))
        center_of_mass = np.zeros((2, 1))
        num_neighbors = len(self.neighbors)

        if num_neighbors > 0:
            for _, agent in self.neighbors.items():
                center_of_mass += agent.pos[:2]
            center_of_mass /= num_neighbors

            # Calculate the direction vector to the center of mass
            direction_vector = center_of_mass / np.linalg.norm(center_of_mass)

            # Calculate the velocity magnitude (proportional to distance to center of mass)
            velocity_magnitude = self.max_velocity * np.linalg.norm(center_of_mass)

            # Calculate the velocity vector
            self.cohesion_vel = velocity_magnitude * direction_vector
        else:
            self.cohesion_vel = np.zeros((2, 1))

        self.cohesion_vel = self.k_coh * self.cohesion_vel

    def migration(self):
        """
        Calculates the migration velocity component.
        """
        if self.migration_target is not None:
            # Calculate direction vector towards the migration target
        
            direction_vector = self.migration_target - self.p_wf[:2] 
            target_d = np.linalg.norm(direction_vector)
            direction_vector = direction_vector / target_d

            # Calculate velocity magnitude (proportional to distance to target)
            velocity_magnitude = self.max_velocity * (target_d**2)
            if target_d >= 0.1:
            # Calculate the migration velocity vector
                self.migration_vel = velocity_magnitude * direction_vector
        else:
            self.migration_vel = np.zeros((2, 1))

        self.migration_vel = self.k_mig * self.migration_vel
       

    def obstacle_avoidance(self):
        # Receives Map
        # Calculate my position within the map
        coord = self.svc.location_to_map(self.pose)
        # Open a window with a certain range in front of the agent
        local_map = self.get_local_neighborhood(coord)
        rotated_local_map = rotate_matrix(local_map, self.p_wf[2])
        visible_local_map = rotated_local_map[:5, :]
        # Find obstacle cells in the local map
        obstacle_indices = np.where(visible_local_map == 100)
        if len(obstacle_indices) ==0:
            return
        self.repulsive_vel = np.zeros((2, 1)) 
        for i, j in zip(obstacle_indices[0], obstacle_indices[1]):
            # Calculate distance to obstacle cell in map coordinates
            distance_map = np.sqrt((i - coord[0])**2 + (j - coord[1])**2)  
            if 0 < distance_map < 3:  # Consider obstacles within a certain radius
                # Calculate direction vector (in map coordinates)
                direction_map = np.array([i - coord[0], j - coord[1]]) 
                direction_map = direction_map / distance_map  # Normalize

                # Calculate velocity magnitude (inversely proportional to square of distance)
                velocity_magnitude = self.max_velocity / (distance_map **2)  

                # Accumulate repulsive velocity (in map coordinates)
                self.repulsive_vel += velocity_magnitude * direction_map.reshape(2, 1)
            
        self.repulsive_vel = self.k_obs * self.repulsive_vel/len(obstacle_indices)

        # If obstacle present, calculate direction to avoid obstacle
        # Transform direction from map coordinates to location
        # Send a velocity toward obstacle avoidance direction
    def get_local_neighborhood(self, coordinate, radius=10):
        """
        Extracts the local neighborhood of a given coordinate in a 2D matrix.

        Args:
            matrix: The 2D numpy matrix.
            coordinate: A tuple (row, col) representing the coordinate.
            radius: The radius of the neighborhood in cells.

        Returns:
            A 2D numpy array representing the local neighborhood.
        """
        matrix = np.array(self.svc.map).reshape(self.svc.map_dim)
        row, col = coordinate
        row_min = max(0, row - radius)
        row_max = min(matrix.shape[0], row + radius + 1)
        col_min = max(0, col - radius)
        col_max = min(matrix.shape[1], col + radius + 1)

        return matrix[row_min:row_max, col_min:col_max]

    

if __name__ == '__main__':
    rospy.init_node('swarm_controller_node')

    num_of_robots = rospy.get_param("/num_of_robots")
    
    swarm = [BoidController(i, num_of_robots) for i in range(num_of_robots)]

    rospy.spin()
