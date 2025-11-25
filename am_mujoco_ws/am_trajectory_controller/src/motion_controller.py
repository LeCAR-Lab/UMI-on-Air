from tqdm import tqdm
import numpy as np
import yaml
import time
import logging
import sys
import copy
import rospy

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from planner.ik_util import ee_pos_to_base_pos_and_arm_joints_4dof
from planner.ee_ik_DH_4dof import init_ik_DH

# Define the global states
from planner.ee_ik_DH_4dof import FREEFLIGHT, TAKEOFF, LAND, PAUSE



logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class MotionPlannerController:
    def __init__(self, config):
        self.current_base_pos = None
        self.current_base_quat = None
        self.arm_joints = None
        self.state_status = None

    
    def update_drone_state(self, drone_pose):
        '''
        input: 
            drone_pose:
                base_x, base_y, base_z, base_quat_x, base_quat_y, base_quat_z, base_quat_w
        output:
        '''
        self.current_base_pos = drone_pose[:3]
        self.current_base_quat = drone_pose[3:7]



    def update_arm_state(self, joint_angles):
        '''
        input: 
            joint_angles: 
                arm_joint_angle[0], arm_joint_angle[1], arm_joint_angle[2], arm_joint_angle[3]
        output:
        '''
        self.arm_joints = joint_angles[:4]

    def update_UAM_status(self, status):
        '''
        input: 
            status: status
        output:
        '''
        self.state_status = status


    def get_cmd(self, target_ee_pose):
        '''
        input: 
            target_ee_pose: [x,y,z, qx,qy,qz,qw]
        output: 
            target_base_pos, 
            reference_base_thrust,
            target_base_ori,
            reference base_torque,
            target_arm_joints,
    
        '''
        raise NotImplementedError("get_cmd() not implemented in base class")
    
    def get_cmd_from_traj(self, target_ee_pose_traj):
        '''
        input: 
            target_ee_pose_traj:
        output: 
            target_base_pos, 
            reference_base_thrust,
            target_base_ori,
            reference base_torque,
            target_arm_joints
        '''
        raise NotImplementedError("get_cmd_from_traj() not implemented in base class")





class IKMotionPlannerController(MotionPlannerController):
    def __init__(self, config):
        super(IKMotionPlannerController, self).__init__(config)

        self.send_vel = config.get("send_vel", False)
        self.send_thrust = config.get("send_thrust", False)
        self.send_arm = config.get("send_arm", True)
                
        self.dt = config.get("dt", 0.05)
        self.rate = rospy.Rate(1 / self.dt)
                        
        # ee_traj: px,py,pz
        geofence = config.get("geofence", [5.0, 3.0, 2.0, -0.05, -3.0, 0.0])
        geofence_max = np.array(geofence[:3])
        geofence_min = np.array(geofence[3:])
        self.geofence = np.array([geofence_max, geofence_min])
        
        
        self.gripper_target = 1.0
        self.rolldof_target = 0.0

        self.arm_joints = np.array([0.0, 0.0, 1.57])     # in radians

        self.state_status = TAKEOFF

        self.ik_planner = init_ik_DH()
        logging.info("IK planner initialized.")
        
        self.IK_time = []



        self.land_arm_joint_angles = np.array([10.0, 10.0, 100])       # in degrees
        self.pause_arm_joint_angles = np.array([10.0, 10.0, 90])      # in degrees 
        self.last_arm_joint_angles = np.array([10.0, 10.0, 90])       # in degrees
        self.last_arm_joint_angles_cmd = np.array([10.0, 10.0, 90])       # in degrees

        # end-effector position constraints
        self.ee_pos_ub = np.array([2.7, 2.0, 2.0])
        self.ee_pos_lb = np.array([-1.0, -2.0, 0.0])
        self.base_pos_ub = np.array([0.5, 2.0, 2.0])
        self.base_pos_lb = np.array([-1.0, -2.0, 0.0])



    def get_reference_state_fulldof(self, 
                                    ik_planner, 
                                    target_ee_pos, 
                                    target_ee_quat, 
                                    current_base_pos, 
                                    current_base_quat, 
                                    arm_joints, 
                                    state_status):
        # print("state status: ", state_status)
        if state_status == LAND:
            target_base_pos = target_ee_pos
            target_base_ori = target_ee_quat
            target_arm_joints = self.land_arm_joint_angles / np.pi * 180.0
            real_target_ee_state = np.zeros(7)
        elif state_status == PAUSE:
            target_base_pos = target_ee_pos
            target_base_ori = target_ee_quat
            target_arm_joints = self.pause_arm_joint_angles / np.pi * 180.0
            real_target_ee_state = np.zeros(7)
        else:
            
            ret = ee_pos_to_base_pos_and_arm_joints_4dof(ik_planner, 
                                                    target_ee_pos,
                                                    target_ee_quat,
                                                    current_base_pos,
                                                    current_base_quat,
                                                    arm_joints,
                                                    state_status)
            if ret is None:
                rospy.logerr_throttle(1, "IK solver failed, use the current base position and joint angles.")
                km_target = np.zeros(11)
                km_target[:3] = current_base_pos
                km_target[3:7] = current_base_quat
                km_target[7:10] = arm_joints
            else:
                km_target = ret

            real_target_ee_state = ik_planner.forward_kinematics(km_target, base_rpy_use = False)
            real_target_ee_pos = real_target_ee_state[:3]
            real_target_ee_quat = real_target_ee_state[3:]
            # Update the base position, quaternion and joint angles
            target_base_pos = km_target[:3]
            # mujoco quat: w,x,y,z -> gazebo quat: x,y,z,w
            target_base_ori = np.array([km_target[4], km_target[5], km_target[6], km_target[3]])
            target_arm_joints = km_target[7:10] / np.pi * 180.0

        return target_base_pos, target_base_ori, target_arm_joints, real_target_ee_state


    
        
    # def solver_fps_timer(self, event):
    #     if len(self.IK_time) > 0:
    #         logging.info("IK solver FPS: {:.2f}".format(1.0/np.mean(self.IK_time)))
    #         self.IK_time = []


    def get_cmd(self, target_ee_pose):
        '''
            input: 
                target_ee_pose: [x,y,z, qx,qy,qz,qw]
            output: 
                target_base_pos, 
                reference_base_thrust,
                target_base_ori,
                reference base_torque,
                target_arm_joints,
        '''


        target_ee_pos = target_ee_pose[:3]
        # target_ee_quat: qx,qy,qz,qw
        target_ee_quat = target_ee_pose[3:]
        for i in range(3):
            if target_ee_pos[i] > self.ee_pos_ub[i]:
                target_ee_pos[i] = self.ee_pos_ub[i]
            elif target_ee_pos[i] < self.ee_pos_lb[i]:
                target_ee_pos[i] = self.ee_pos_lb[i]

        # Get the target base position and joint angles from the IK planner
        current_base_pos = self.current_base_pos
        current_base_quat = self.current_base_quat
        current_base_quat = np.array([0.0, 0.0, 0.0, 1.0])

        base_pos_ub = self.base_pos_ub
        base_pos_lb = self.base_pos_lb
        self.ik_planner.base_constrain_callback(base_pos_ub, base_pos_lb)
        
        target_base_pos, target_base_ori, target_arm_joints, real_target_ee_state = self.get_reference_state_fulldof(self.ik_planner,
                                                                                                target_ee_pos,
                                                                                                target_ee_quat,
                                                                                                current_base_pos,
                                                                                                current_base_quat,
                                                                                                self.arm_joints,
                                                                                                self.state_status)
        

        if np.linalg.norm(target_arm_joints - self.last_arm_joint_angles) <=40:
            self.last_arm_joint_angles = copy.deepcopy(target_arm_joints)
        else:
            target_arm_joints = copy.deepcopy(self.last_arm_joint_angles_cmd[:3])
        
        # if np.linalg.norm(target_arm_joints - self.last_arm_joint_angles) <=90:
        #     self.last_arm_joint_angles = target_arm_joints
        # else:
        #     target_arm_joints = self.last_arm_joint_angles

        reference_base_thrust = np.zeros(3)
        reference_base_torque = np.zeros(3)


        return target_base_pos, reference_base_thrust, target_base_ori, reference_base_torque, target_arm_joints