#!/usr/bin/env python3

import numpy as np

# ROS
import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

class GripperClient:
    def __init__(self, namespace, side):
        TOPIC_NAME = namespace + "/" + side + '_gripper_controller/command'
        self.publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)
        while self.publisher.get_num_connections() == 0:
            rospy.sleep(1)
            
        # slipper
        # self.gripper_open_pose = np.array([0.09]) # meter
        # self.gripper_close_pose = np.array([0.03])

        # apron
        self.gripper_open_pose = np.array([0.01]) # meter
        self.gripper_close_pose = np.array([0.00])
            
        # Create JointTrajectory message
        self.trajectory = self.__create_joint_trajectory_base()
        self.trajectory.joint_names = [
                side + '_gripper/finger_joint'
                ]
        
    def __command(self, pose):
        self.trajectory.header.stamp = rospy.Time.now()
        self.trajectory.points[0].positions = pose.tolist()
        self.publisher.publish(self.trajectory)

    def __create_joint_trajectory_base(self):
        # Creates a message.
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Time(0)
        trajectory.points.append(point)

        return trajectory 
    
    def open_gripper(self):
        self.__command(self.gripper_open_pose)

    def close_gripper(self):
        self.__command(self.gripper_close_pose)

if __name__ == "__main__":    

    rospy.init_node('gripper_controller', anonymous=True, disable_signals=True)
