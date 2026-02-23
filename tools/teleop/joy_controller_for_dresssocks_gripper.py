#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# インピーダンスの代わりにyawを追加

import threading
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory
import numpy as np
import tf
import time
from robot import Torobo
from gripper_controller_for_dresssocks import GripperClient
from publish_joint_trajectory import publish_joint_trajectory
from joy_impedance_adjuster import ImpedanceAdjuster

# TOPIC_NAME_position = '/torobo/joint_trajectory_controller/command'
# TOPIC_NAME_impedance = '/torobo/joint_trajectory_controller/command'
TOPIC_NAME_position = '/torobo/online_joint_trajectory_controller/command'
TOPIC_NAME_impedance = '/torobo/online_joint_impedance_controller/command'
# # TOPIC_NAME_position = '/torobo/online_joint_upperarm_trajectory_controller/command'
# # TOPIC_NAME_impedance = '/torobo/online_joint_lowerarm_impedance_controller/command'

POS_JOINT_NUM = 7
IMP_JOINT_NUM = 1
RATE_FILTER = 0.30

class JoyController(Torobo):
    def __init__(self, freq, movegroup="left_arm", tooltip="left_arm/link_tip", side="left"):
        super().__init__(movegroup=movegroup, tooltip=tooltip)
        
        self.side = side
        self.freq = freq
    
        self._xyz_diff = np.zeros(3)
        self._z_movement = 0.0
        self._scale_xyz = [0.003, 0.003, 0.001] 

        self._rpy_diff = np.zeros(3)
        self._scale_rpy = [0.01, 0.01, 0.01] # slipper:0.02, apron:0.005
        self._position_list = []
        
        self._parallel_diff = np.zeros(3)
        self._right_parallel_x_movement = 0.0
        self._right_parallel_y_movement = 0.0
        self._right_parallel_z_movement = 0.0
        self._left_parallel_x_movement  = 0.0
        self._left_parallel_y_movement  = 0.0
        self._left_parallel_z_movement  = 0.0
        
        self.gripper_state = 0
        self.arm_t = None
        self.res_previous = None
        self.joy_time = 0.0
        self.joy_duration = 0.0
        self.is_joint_initialize = False
        
        self.joint_pos_msg = JointState()
        self.joint_imp_msg = JointState()
        self.joint_pos_msg.name=[self.side + "_arm/joint_"+ str(i+1) for i in range(POS_JOINT_NUM) ]
        self.joint_imp_msg.name=[self.side + "_arm/joint_"+ str(i+5) for i in range(IMP_JOINT_NUM) ]
        
        self.gripper_msg = JointState()
        self.gripper_msg.name = [self.side + "/state"]
        
        self.gripper = GripperClient("/torobo", self.side)
        self.gripper_pose = Pose()
        
        self.lock = threading.Lock()
        
        self.impedande = ImpedanceAdjuster()
        self.is_impedance_set = True
    
        rospy.Subscriber("/joy", Joy, self.__joy_callback)
        rospy.Subscriber('/torobo/joint_states', JointState, self.__joint_callback)

        self.joint_pub_pos = rospy.Publisher(TOPIC_NAME_position, JointTrajectory, queue_size=1)
        self.joint_pub_imp = rospy.Publisher(TOPIC_NAME_impedance, JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper_states", JointState, queue_size=1)
        self.gripper_pose_pub = rospy.Publisher("/gripper_pose", Pose, queue_size=1)
        
        rospy.wait_for_message('/torobo/joint_states', JointState, timeout=None)
        self.init_xyzrpy = np.array(self.compute_fk(joint_angles=self.arm_t))

    def __joy_callback(self, msg):
        with self.lock:
            # print(self.gripper_state)
            #  Check if there is any movement or button press
            if np.any(np.array([
                msg.axes[0], msg.axes[1],
                msg.axes[3], msg.axes[4], 
                msg.axes[6], msg.axes[7],
                msg.buttons[2], msg.buttons[3],
                msg.buttons[4], msg.buttons[5], 
                msg.buttons[6], msg.buttons[7]
                ]) != 0):
                self.joy_time = time.time()
                
            gripper_cmd = np.array(msg.buttons[0:2])
            if self.gripper_state == 0 and gripper_cmd[0]:
                self.gripper.close_gripper()
                self.gripper_state = 1
                print("gripper close")

            if self.gripper_state == 1 and gripper_cmd[1]:
                self.gripper.open_gripper()
                self.gripper_state = 0
                print("gripper open")
            
            # change_impedance
            # impedance_cmd = np.array(msg.buttons[2:4])
            # if impedance_cmd[0] and self.is_impedance_set == True:
            #     self.is_impedance_set = False
            #     self.impedande.main("u")

            # if impedance_cmd[1] and self.is_impedance_set == True:
            #     self.is_impedance_set = False
            #     self.impedande.main("d")
            
            # buttons[4]:L1 buttons[5]:R1 up
            # buttons[6]:L2 buttons[7]:R2 down
            if msg.buttons[4]:
                self._z_movement  = 1.0
            elif msg.buttons[5]:
                self._right_parallel_z_movement =  1.0
                self._left_parallel_z_movement  = -1.0 
            elif msg.buttons[6]:
                self._z_movement  = -1.0
            elif msg.buttons[7]:
                self._right_parallel_z_movement = -1.0
                self._left_parallel_z_movement  =  1.0 
            else:
                self._z_movement = 0.0
                self._right_parallel_z_movement = 0.0
                self._left_parallel_z_movement  = 0.0 
                
            if self.side == "right":
                self._right_parallel_x_movement = msg.axes[4] 
                self._right_parallel_y_movement = msg.axes[3] 
                self._parallel_diff = np.array([self._right_parallel_x_movement, self._right_parallel_y_movement, self._right_parallel_z_movement]) * self._scale_xyz
                
            elif self.side == "left":
                self._left_parallel_x_movement = -msg.axes[4] 
                self._left_parallel_y_movement = -msg.axes[3] 
                self._parallel_diff = np.array([self._left_parallel_x_movement, self._left_parallel_y_movement, self._left_parallel_z_movement]) * self._scale_xyz

            self._xyz_diff = np.array([msg.axes[1], msg.axes[0], self._z_movement]) * self._scale_xyz
            
            roll_input = 0.0
            pitch_input = 0.0
            yaw_input = 0.0

            # Roll: D-pad vertical
            if msg.axes[7] == 1.0:
                roll_input = 1.0
            elif msg.axes[7] == -1.0:
                roll_input = -1.0

            # Pitch: D-pad horizontal
            if msg.axes[6] == 1.0:
                pitch_input = 1.0
            elif msg.axes[6] == -1.0:
                pitch_input = -1.0

            # Yaw: △/□
            triangle_pressed = msg.buttons[2]
            square_pressed = msg.buttons[3]

            if triangle_pressed:
                yaw_input = 1.0
            elif square_pressed:
                yaw_input = -1.0
            
            if self.side == "right":
                roll_input = -roll_input
                yaw_input = -yaw_input

            self._rpy_diff = np.array([roll_input, pitch_input, yaw_input]) * self._scale_rpy
        
    def __joint_callback(self, msg):
        # print(self.side)
        with self.lock:
            if self.side == "left":
                self.arm_t = np.array(msg.position[10:17]) # check
                self.is_joint_initialize = True
            elif self.side == "right":
                self.arm_t = np.array(msg.position[19:26]) # check
                self.is_joint_initialize = True

    def run(self):
        r = rospy.Rate(self.freq)
        
        while not rospy.is_shutdown():
            with self.lock:
                if self.arm_t is None or self._xyz_diff is None or self._parallel_diff is None:
                    continue
                self._xyzrpy = np.array(self.compute_fk(joint_angles=self.arm_t))
                
                if np.any(self._xyz_diff != 0):
                    self._xyzrpy[:3] += self._xyz_diff
                if np.any(self._parallel_diff != 0):
                    self._xyzrpy[:3] += self._parallel_diff
                if np.any(self._rpy_diff != 0):
                    self._xyzrpy[3:] += self._rpy_diff
                
                self.gripper_pose.position.x = self._xyzrpy[0]
                self.gripper_pose.position.y = self._xyzrpy[1]
                self.gripper_pose.position.z = self._xyzrpy[2]
                
                x, y, z, w = tf.transformations.quaternion_from_euler(self._xyzrpy[3], self._xyzrpy[4], self._xyzrpy[5])
                self.gripper_pose.orientation.x = x
                self.gripper_pose.orientation.y = y
                self.gripper_pose.orientation.z = z
                self.gripper_pose.orientation.w = w
            
                self.res = self.compute_ik(
                    x=self._xyzrpy[0], y=self._xyzrpy[1], z=self._xyzrpy[2],
                    roll=self._xyzrpy[3], pitch=self._xyzrpy[4], yaw=self._xyzrpy[5],
                    joint_angles=self.arm_t
                )
                
                ik_success = False
                
                if self.res is None:
                    rospy.logwarn("Skip joint update because IK returned no solution")
                else:
                    if self.side == "left" and self.is_joint_initialize == True:
                        self.joint_pos_msg.position = self.arm_t[0:POS_JOINT_NUM]*RATE_FILTER + np.array(self.res.position[9:9+POS_JOINT_NUM])*(1-RATE_FILTER) # check
                        self.is_joint_initialize = False
                        ik_success = True
                    elif self.side == "right" and self.is_joint_initialize == True:
                        self.joint_pos_msg.position = self.arm_t[0:POS_JOINT_NUM]*RATE_FILTER + np.array(self.res.position[18:18+POS_JOINT_NUM])*(1-RATE_FILTER) # check
                        self.is_joint_initialize = False
                        ik_success = True
                    
                self.joy_duration = time.time() - self.joy_time
                if ik_success and self.joy_duration < 1.0 and (np.any(self._xyz_diff != 0) or np.any(self._parallel_diff != 0) or np.any(self._rpy_diff != 0)):
                    print(f"{self.side}:in")
                    publish_joint_trajectory(
                    publisher =  self.joint_pub_pos,
                    joint_names = self.joint_pos_msg.name,
                    positions =self.joint_pos_msg.position,
                    time_from_start = 1.0 )
                    # publish_joint_trajectory(
                    # publisher =  self.joint_pub_imp,
                    # joint_names = self.joint_imp_msg.name,
                    # positions =self.joint_imp_msg.position,
                    # time_from_start = 1.0 )
                
            r.sleep()

def main(freq):
    
    controllers = [
        JoyController(freq=freq, movegroup="right_arm", tooltip="right_arm/link_tip", side="right"),
        JoyController(freq=freq, movegroup="left_arm", tooltip="left_arm/link_tip", side="left"),
    ]
    
    threads = []
    for controller in controllers:
        thread = threading.Thread(target=controller.run)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    try:
        rospy.init_node('joy_control_node', anonymous=True)
        main(freq=150)
    except rospy.ROSInterruptException:
        pass
    
