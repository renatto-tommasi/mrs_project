#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
from mrs_project.msg import OdometryList


class Communications:
    def __init__(self, num_of_robots):   

        self.odometry_list = [None] * num_of_robots 

        self.odometry_list_pub = rospy.Publisher("/swarm_odometry", OdometryList, queue_size=10)

        for i in range(num_of_robots):
            rospy.Subscriber("/robot_{}/odom".format(i), Odometry, self.get_odometry, callback_args=i)

    def get_odometry(self, msg, robot_id):
        # Update the odometry list for the specific robot
        self.odometry_list[robot_id] = msg
        # rospy.loginfo(f"Updated odometry for robot {robot_id}: {msg.pose.pose.position}")

        # After updating the list, publish the OdometryList message
        self.publish_odometry_list()

    def publish_odometry_list(self):
        # Create an OdometryList message
        odometry_list_msg = OdometryList()
        odometry_list_msg.odometry_list = [None] * num_of_robots

        # Add each odometry message to the OdometryList message
        for i , odom in enumerate(self.odometry_list):

            odometry_list_msg.odometry_list[i] = odom

        # Publish the OdometryList message
        # rospy.loginfo(f"Publishing OdometryList with {len(odometry_list_msg.odometry_list)} odometry messages.")
        self.odometry_list_pub.publish(odometry_list_msg)


if __name__ == '__main__':
    rospy.init_node('communication_node')
    rospy.loginfo(f"Communication Node Initialized")


    num_of_robots = rospy.get_param("/num_of_robots")

    node = Communications(num_of_robots)

    rospy.spin()