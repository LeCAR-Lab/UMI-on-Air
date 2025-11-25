#!/home/xiaofeng/anaconda3/envs/uamteleop/bin/python
##!/usr/bin/python3


import numpy as np
from tqdm import tqdm
import numpy as np
import yaml
import copy
import time
import logging


import rospy
from std_msgs.msg import Bool, Float32, Float64MultiArray, MultiArrayDimension
from geometry_msgs.msg import Quaternion, Point, Vector3
from sensor_msgs.msg import JointState


from nav_msgs.msg import Odometry
from core_pose_controller.msg import PoseCtrlTarget

from behavior_tree_msgs.msg import BehaviorTreeCommands
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from planner.ik_util import ee_pos_to_base_pos_and_arm_joints_4dof
from planner.ee_ik_DH_4dof import init_ik_DH
from copy import deepcopy
from planner.ee_mpc_acado_4dof import ArmMPCPlanner
from planner.ee_mpc_acado import DisturbanceObserver

# Define the global states
from planner.ee_ik_DH_4dof import WAITING, FREEFLIGHT, TAKEOFF, LAND, PAUSE



def quaternion_to_euler(quat):
    # quat: w,x,y,z
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]
    r = R.from_quat([qx, qy, qz, qw])
    euler = r.as_euler('zyx', degrees=False)
    euler = euler[::-1]
    return euler

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

def xyzw_to_wxyz(quat):
    return np.concatenate((quat[:, 3:], quat[:, 0:3]), axis=1)


class TrajectorySender:
    def __init__(self):
        config_path = rospy.get_param("~params", {})
        xml_path = rospy.get_param("~xml_path", "hexa_scorpion.xml")
        assert config_path, "No configuration found."
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        rospy.loginfo(f"Configuration loaded from: {config_path}")

        self.send_vel = config.get("send_vel", True)
        self.send_thrust = config.get("send_thrust", False)
        self.send_arm = config.get("send_arm", False)
        rospy.loginfo(f"Topic sending: vel: {self.send_vel}, thrust: {self.send_thrust}, arm: {self.send_arm}")
                
        # self.dt = config.get("dt", 0.01)
        self.dt = 0.01
        self.rate = rospy.Rate(1 / self.dt)
        self.odom = None

        self.traj_state_dim = 7
        self.ee_traj = np.empty((0, self.traj_state_dim))

        


        # ee_traj: px,py,pz
        geofence = config.get("geofence", [5.0, 3.0, 2.0, -0.05, -3.0, 0.0])
        geofence_max = np.array(geofence[:3])
        geofence_min = np.array(geofence[3:])
        self.geofence = np.array([geofence_max, geofence_min])
        
        self.return_home_pos = config.get("return_home_pos", [0.0, 0.0, 1.25])
        self.return_home_quat = config.get("return_home_quat", [0.0, 0.0, 0.0, 1.0])
        self.return_home_pos = np.array(self.return_home_pos)
        self.return_home_quat = np.array(self.return_home_quat)
        
    

        self.t_bar = tqdm(total=len(self.ee_traj))
        self.takeoff = False

        # This is the default initial position the drone before takeoff, the ground_z is from this variable
        self.origin = config.get("origin", [0.0, 0.0, 0.0])


        self.takeoff_duration = config.get("takeoff_duration", 8.0)
        self.takeoff_duration = 8.0
        self.land_duration = config.get("land_duration", 8.0)
        
        
        self.traj_pub = rospy.Publisher(
            "/tracking_target", PoseCtrlTarget, queue_size=5
        )
        self.ee_traj_pub = rospy.Publisher("/ee_tracking_target", PoseCtrlTarget, queue_size=5)


        # publish the real final ee tracking point
        self.real_ee_tracking_target_pub = rospy.Publisher("/real_ee_tracking_target", PoseCtrlTarget, queue_size=5)

        # publish the current ee state
        self.ee_state_pub = rospy.Publisher("/ee_state", PoseCtrlTarget, queue_size=5)




     
        self.arm_pub = rospy.Publisher(
            "/manipulator_arm_command", JointState, queue_size=10
        )

        self.odom_sub = rospy.Subscriber(
            "/odometry", Odometry, self.odom_callback, queue_size=10
        )
        self.commander_sub = rospy.Subscriber(
            "/behavior_tree_commands",
            BehaviorTreeCommands,
            self.command_callback,
            queue_size=10,
        )

        # Subscribe to the arm joint states
        self.arm_joint1_sub = rospy.Subscriber(
            "/hexa_scorpion/manipulator_arm1_joint/angle", Float32, self.arm_joint1_callback, queue_size=10
        )
        self.arm_joint2_sub = rospy.Subscriber(
            "/hexa_scorpion/manipulator_arm2_joint/angle", Float32, self.arm_joint2_callback, queue_size=10
        )
        self.arm_joint3_sub = rospy.Subscriber(
            "/hexa_scorpion/manipulator_arm3_joint/angle", Float32, self.arm_joint3_callback, queue_size=10
        )

        self.arm_joint_sub = rospy.Subscriber(
            "/arm_state", JointState, self.arm_state_callback, queue_size=10
        )



        # Subscribe to the online ee target position
        # self.ee_target_sub = rospy.Subscriber("/hexa_scorpion/ee_target", PoseCtrlTarget, self.ee_target_callback, queue_size=10)
        self.ee_target_sub = rospy.Subscriber("/policy/ee_target", PoseCtrlTarget, self.ee_target_callback, queue_size=10)
        
        self.raw_gripper_target = 1.0
        self.safe_gripper_target = 1.0
        self.ee_gripper_target_sub = rospy.Subscriber("/policy/ee_gripper_target", Float32, self.ee_gripper_target_callback, queue_size=5)

        self.rolldof_target = 0.0
        self.rolldof_target_sub = rospy.Subscriber("/policy/ee_rolldof_target", Float32, self.rolldof_target_callback, queue_size=5)

        # MPC horizon (policy-provided EE horizon) and precedence bookkeeping
        self.horizon_buf = None  # shape (N,7) pos[3], quat_xyzw[4]
        self._horizon_time = None
        self._ik_target_time = None
        self.ee_horizon_sub = rospy.Subscriber(
            "/ee_trajectory_target", Float64MultiArray, self.ee_horizon_callback, queue_size=5
        )



        self.arm_joints = np.array([0.0, 0.0, 1.57])     # in radians

    
        # self.initial_ee_offset = np.array([0.0746119517, -0.000115239730, 0.1])
        # self.takeoff_ee_pos = self.takeoff_base_pos + self.initial_ee_offset
        self.after_takeoff_ee_pos_target = np.array([0.0, 0.0, 1.3])

        self.takeoff_quat = np.array([0.0, 0.0, 0.0, 1.0])
        
        self.state_status = TAKEOFF


        self.ik_planner = init_ik_DH()
        rospy.loginfo("IK planner initialized.")

        # MPC planner configuration (kept unused unless horizon is newer than IK)
        mpc_params = yaml.safe_load(open(config_path, "r"))
        mpc_params = mpc_params.get("mpc", {})
        if len(mpc_params) == 0:
            rospy.logwarn("No MPC parameters provided in config; MPC horizon will be ignored.")
        else:
            mass = mpc_params['mass']
            T = mpc_params['T']
            N = mpc_params['N']
            Q = mpc_params['Q']
            R = mpc_params['R']
            R_arm_delta = mpc_params['R_delta']
            acc2thrust_gain = mpc_params['acc2thrust_gain']
            pos_min = mpc_params['pos_min']
            pos_max = mpc_params['pos_max']
            vel_min = mpc_params['vel_min']
            vel_max = mpc_params['vel_max']
            acc_min = mpc_params['acc_min']
            acc_max = mpc_params['acc_max']
            joint_min = mpc_params['joint_min']
            joint_max = mpc_params['joint_max']
            default_arm_angle = mpc_params['default_arm_angle']
            output_filter_gain = mpc_params['output_filter_gain']
            moment_of_inertia = mpc_params['moment_of_inertia']

            self.planner = ArmMPCPlanner(
                mass, T, N, Q, R, R_arm_delta, acc2thrust_gain,
                pos_min, pos_max, vel_min, vel_max, acc_min, acc_max,
                joint_min, joint_max, default_arm_angle, output_filter_gain, moment_of_inertia
            )

            dob_params = yaml.safe_load(open(config_path, "r")).get("disturbance_observer", {})
            if len(dob_params) > 0:
                self.dob = DisturbanceObserver(
                    cutoff_freq=dob_params["cutoff_freq"],
                    acc2thrust_gain=mpc_params["acc2thrust_gain"],
                    dt=self.dt,
                    acc_min=dob_params["acc_min"],
                    acc_max=dob_params["acc_max"],
                )
                self.enable_dob = dob_params.get("enable", False)
            else:
                self.dob = None
                self.enable_dob = False

        self.IK_time = []
        
        self.timer = rospy.Timer(rospy.Duration(1.0), self.solver_fps_timer)

        self.land_arm_joint_angles = np.array([10.0, 10.0, 90])       # in degrees
        self.pause_arm_joint_angles = np.array([10.0, 10.0, 90])      # in degrees 
        self.last_arm_joint_angles = np.array([10.0, 10.0, 90])       # in degrees
        self.last_arm_joint_angles_cmd = np.array([10.0, 10.0, 90])       # in degrees


        # end-effector position constraints
        self.ee_pos_ub = np.array([2.3, 1.0, 1.8])
        self.ee_pos_lb = np.array([-1.0, -1.0, 0.0])


        self.base_pos_ub = np.array([1.2, 1.4, 2.0])
        self.base_pos_lb = np.array([-1.0, -1.4, 0.0])


        self.obj_in_the_hand = True
        self.gripper_open_cnt = 0


        while self.odom is None:
            rospy.logwarn_throttle(1, "No odometry received.")
            rospy.sleep(1)

        self.before_takeoff_init()

        
    def before_takeoff_init(self):
        rospy.sleep(1)
        self.before_takeoff_ee_state = copy.deepcopy(self.ee_state)
        self.before_takeoff_ee_pos = self.before_takeoff_ee_state[:3]
        self.before_takeoff_base_state = copy.deepcopy(self.base_state)
        self.before_takeoff_base_pos = self.before_takeoff_base_state[:3]
        self.before_takeoff_arm_joints = copy.deepcopy(self.arm_joints)



    def solver_fps_timer(self, event):
        if len(self.IK_time) > 0:
            rospy.loginfo("IK solver FPS: {:.2f}".format(1.0/np.mean(self.IK_time)))
            self.IK_time = []

    def rolldof_target_callback(self, msg):
        self.rolldof_target = msg.data

    def odom_callback(self, msg):
        self.odom = msg

        # calculate the ee position and quaternion
        base_pos = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        base_quat = np.array(
            [
                self.odom.pose.pose.orientation.w,
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
            ]
        )
        base_state = np.concatenate([base_pos, base_quat])
        self.base_state = copy.deepcopy(base_state)

        base_euler = quaternion_to_euler(base_quat)
        manipulator_arm_angles = np.array([self.arm_joints[0], self.arm_joints[1], self.arm_joints[2]])
        u = np.concatenate([base_pos, base_euler, manipulator_arm_angles])
        ee_state = self.ik_planner.forward_kinematics(u)
        ee_pos = ee_state[:3]
        ee_quat = ee_state[3:]
        self.ee_state = copy.deepcopy(ee_state)
        ee_state_msg = PoseCtrlTarget()
        ee_state_msg.header.stamp = rospy.Time.now()
        ee_state_msg.pose.position = Point(*ee_pos)
        ee_state_msg.pose.orientation.w = ee_quat[0]
        ee_state_msg.pose.orientation.x = ee_quat[1]
        ee_state_msg.pose.orientation.y = ee_quat[2]
        ee_state_msg.pose.orientation.z = ee_quat[3]
        self.ee_state_pub.publish(ee_state_msg)


    def arm_state_callback(self, msg):
        current_arm_joint_angles = np.array(msg.position)
        self.arm_joints[0] = current_arm_joint_angles[0] * np.pi / 180.0
        self.arm_joints[1] = current_arm_joint_angles[1] * np.pi / 180.0
        self.arm_joints[2] = (current_arm_joint_angles[2]) * np.pi / 180.0


    def arm_joint1_callback(self, msg):
        self.arm_joints[0] = msg.data
    
    def arm_joint2_callback(self, msg):
        self.arm_joints[1] = msg.data
    
    def arm_joint3_callback(self, msg):
        self.arm_joints[2] = msg.data

    def command_callback(self, msg):
        for command in msg.commands:
            if command.condition_name == "Takeoff Commanded" and command.status == 2:
                rospy.loginfo("Takeoff command received.")
                self.takeoff_callback(command)
                
            elif command.condition_name == "Land Commanded" and command.status == 2:
                rospy.loginfo("Land command received.")
                self.land_callback(command)
                
            elif command.condition_name == "Pause Commanded" and command.status == 2:
                rospy.loginfo("Pause command received.")
                self.pause_callback(command)
                
            elif (
                command.condition_name == "Return Home Commanded"
                and command.status == 2
            ):
                rospy.loginfo("Return home command received.")
                self.returnhome_callback(command)

    def ee_gripper_target_callback(self, msg):
        self.raw_gripper_target = msg.data




    def ee_target_callback(self, msg):
        if not self.takeoff:
            rospy.logwarn("Not taking off, cannot receive ee target.")
            return
        if self.state_status == TAKEOFF:
            rospy.logwarn("During takeoff, cannot receive ee target.")
            return
        # TODO: add reference model to avoid sudden jump
        ee_target_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        ee_target_quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        ee_target_state = ee_target_pos + ee_target_quat


        self.traj = [self.traj[0], ee_target_state]
        # Timestamp for IK precedence vs MPC horizon
        self._ik_target_time = rospy.Time.now()

    def ee_horizon_callback(self, msg: Float64MultiArray):
        data = np.array(msg.data, dtype=float)
        # Ensure shape (N,7)
        N = None
        D = None
        if hasattr(msg, "layout") and len(msg.layout.dim) >= 2:
            N = msg.layout.dim[0].size
            D = msg.layout.dim[1].size
        if N is None or D is None:
            if data.size % 7 != 0:
                rospy.logwarn_throttle(1.0, f"Invalid EE horizon size {data.size}, not divisible by 7. Ignoring.")
                return
            N = data.size // 7
            D = 7
        if D != 7:
            rospy.logwarn_throttle(1.0, f"Invalid EE horizon action dim {D}, expected 7. Ignoring.")
            return
        horizon = data.reshape((N, 7))
        # Normalize quaternions; if near-zero, set to identity
        quat = horizon[:, 3:7]
        norms = np.linalg.norm(quat, axis=1)
        near_zero = norms < 1e-6
        if np.any(near_zero):
            quat[near_zero] = np.array([0.0, 0.0, 0.0, 1.0])
            norms[near_zero] = 1.0
        quat = quat / norms[:, None]
        horizon[:, 3:7] = quat
        self.horizon_buf = horizon
        self._horizon_time = rospy.Time.now()

    
    def takeoff_callback(self, msg):
        if self.takeoff:
            rospy.logerr("Already taking off, cannot take off again.")
            return

        takeoff_traj = self.takeoff_planner()
        self.takeoff = True
        self.state_status = TAKEOFF
        self.traj = np.vstack([takeoff_traj, self.ee_traj])

        self.t_bar = tqdm(total=self.traj.shape[0])

        rospy.loginfo("Starting takeoff sequence.")


    def land_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot land.")
            return

        self.traj = list(self.land_planner())
        self.state_status = LAND

        self.t_bar = tqdm(total=len(self.traj))
        rospy.loginfo("Starting landing sequence.")

    

    def returnhome_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot return home.")
            return

        cur_ee_pos = np.array(
            [
                self.traj[0][0],
                self.traj[0][1],
                self.traj[0][2],
            ]
        )
        cur_ee_quat = np.array(
            [
                self.traj[0][3],
                self.traj[0][4],
                self.traj[0][5],
                self.traj[0][6],
            ]
        )
        # Interpolate the position
        tar_pos = self.return_home_pos
        start_pos = cur_ee_pos.reshape((3, 1))
        end_pos = tar_pos.reshape((3, 1))
        pos_traj = self.cos_interp(start_pos, end_pos, self.land_duration)
        # Interpolate the quaternion
        tar_quat = self.return_home_quat
        start_quat = cur_ee_quat
        end_quat = tar_quat
        quat_traj = self.slerp_interpolation(start_quat, end_quat, self.land_duration)
        # Combine the position and quaternion trajectory
        pose_traj = np.hstack([pos_traj, quat_traj])
        state_dim = 7
        traj_dim = (len(pose_traj), state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pose_traj
        self.traj = list(traj)
        self.t_bar = tqdm(total=len(self.traj))
        rospy.loginfo("Returning home.")




    def takeoff_planner(self):
        if self.odom is None:
            rospy.logwarn("No odometry received.")
            return False

        start_base_pos = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        start_base_quat = np.array(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )
        # Interpolate the position
        start_ee_pos = copy.deepcopy(self.ee_state[:3])
        target_ee_pos = copy.deepcopy(self.after_takeoff_ee_pos_target)
        start_pos = start_ee_pos.reshape((3, 1))
        end_pos = target_ee_pos.reshape((3, 1))
        pos_traj = self.cos_interp(start_pos, end_pos, self.takeoff_duration)
        # Interpolate the quaternion
        start_ee_quat = start_base_quat
        target_ee_quat = self.takeoff_quat
        # print("start_ee_quat: ", start_ee_quat)
        # print("target_ee_quat: ", target_ee_quat)
        quat_traj = self.slerp_interpolation(start_ee_quat, target_ee_quat, self.takeoff_duration)
        # Combine the position and quaternion trajectory
        pose_traj = np.hstack([pos_traj, quat_traj])
        
        state_dim = 7
        traj_dim = (pose_traj.shape[0], state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pose_traj
        return traj

    def land_planner(self):
        # For landing, we save the drone base reference trajectory in the "ee_traj"


        if self.odom is None:
            rospy.logwarn("No odometry received.")
            return False
        
        start_base_pos = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        start_base_quat = np.array(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )

        # Interpolate the position
        ground_z = self.origin[2]
        target_base_pos = np.array([start_base_pos[0], start_base_pos[1], ground_z])
        start_pos = start_base_pos.reshape((3, 1))
        end_pos = target_base_pos.reshape((3, 1))
        pos_traj = self.cos_interp(start_pos, end_pos, self.land_duration)
        # Interpolate the quaternion
        
        target_base_quat = start_base_quat   
        quat_traj = self.slerp_interpolation(start_base_quat, target_base_quat, self.land_duration)

        # Combine the position and quaternion trajectory
        pose_traj = np.hstack([pos_traj, quat_traj])
        
        state_dim = 7
        traj_dim = (pose_traj.shape[0], state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pose_traj

        self.land_arm_joint_angles = copy.deepcopy(self.arm_joints)
        print("landing!!!")
        return traj

    def get_base_pose(self):
        if self.odom is None:
            rospy.logwarn("No odometry received.")
            return False
        cur_base_pose = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        cur_base_quat = np.array(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )
        return cur_base_pose, cur_base_quat

    def line_interp(self, start, end, duration):
        t = np.linspace(0, duration, int(duration / self.dt)).reshape((1, -1))
        return (start + (end - start) * t / duration).T
    
    def cos_interp(self, start, end, duration):
        t = np.linspace(0, duration, int(duration / self.dt)).reshape((1, -1))
        return (start + (end - start) @ (1 - np.cos(np.pi * t / duration)) / 2.0).T
    
    def slerp_interpolation(self, start, end, duration):
        t = np.linspace(0, duration, int(duration / self.dt))
        # Create a SLERP object for interpolation between the two rotations
        slerp = Slerp([0, duration], R.from_quat([start, end]))
        # Perform interpolation for the given t values and return the result as quaternions
        interpolated_rots = slerp(t)
        # Convert the interpolated rotations back to quaternions and return them
        return interpolated_rots.as_quat()

    def get_reference_state_fulldof(self, 
                                    ik_planner, 
                                    target_ee_pos, 
                                    target_ee_quat, 
                                    current_base_pos, 
                                    current_base_quat, 
                                    arm_joints, 
                                    state_status):
        
        if state_status == LAND:
            target_base_pos = target_ee_pos
            target_base_ori = target_ee_quat
            target_arm_joints = self.land_arm_joint_angles / np.pi * 180.0
            real_target_ee_state = np.zeros(7)
        elif state_status == PAUSE:
            target_base_pos = target_ee_pos
            target_base_quat = target_ee_quat
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

    def get_third_pitch_fake(self, target_ee_quat_xyzw):
        target_ee_quat_wxyz = np.array([target_ee_quat_xyzw[3], target_ee_quat_xyzw[0], target_ee_quat_xyzw[1], target_ee_quat_xyzw[2]])
        target_euler = quaternion_to_euler(target_ee_quat_wxyz)
        pitch = target_euler[1]
        pitch_degree = pitch * 180.0 / np.pi
        pitch_degree = np.clip(pitch_degree, -30, 30)
        return pitch_degree


    def ee_target_safe_filter(self, target_ee_pos, target_ee_ori):
        current_ee_state = copy.deepcopy(self.ee_state)
        current_ee_pos = current_ee_state[:3]

        current_base_state = copy.deepcopy(self.base_state)
        current_base_pos = current_base_state[:3]

        if np.linalg.norm(target_ee_pos - current_ee_pos) > 0.5:
            safe_target_ee_pos = current_ee_pos + 0.5 * (target_ee_pos - current_ee_pos) / np.linalg.norm(target_ee_pos - current_ee_pos)
        else:
            safe_target_ee_pos = target_ee_pos
        
        if np.abs(target_ee_pos[0] - current_ee_pos[0]) > 0.5:
            safe_target_ee_pos[0] = current_ee_pos[0] + 0.5 * np.sign(target_ee_pos[0] - current_ee_pos[0])

        if np.abs(target_ee_pos[1] - current_ee_pos[1]) > 0.2:
            safe_target_ee_pos[1] = current_ee_pos[1] + 0.2 * np.sign(target_ee_pos[1] - current_ee_pos[1])        

        # make the target_ee_ori = [0, 0, 0, 1]
        safe_target_ee_ori = np.array([0.0, 0.0, 0.0, 1.0])

        gripper_target = copy.deepcopy(self.raw_gripper_target)
        
        # check if it is ok to open the gripper
        # gripper_target = 0.0 means open the gripper
        # gripper_target = 1.0 means close the gripper

        # for peg in hole
        if not self.obj_in_the_hand:
            print("no object in the hand")
            safe_gripper_target = copy.deepcopy(gripper_target)
        else:
            safe_gripper_target = copy.deepcopy(gripper_target)
            if gripper_target < 0.9:
                if current_ee_pos[0]<1.7:
                    safe_gripper_target = 1.0
                    print("ee to close")
                    self.gripper_open_cnt = 0
                if current_ee_pos[0] - current_base_pos[0] < 0.6:
                    print("ee to close to the base")
                    safe_gripper_target = 1.0
                    self.gripper_open_cnt = 0
            if gripper_target < 0.3:
                self.gripper_open_cnt += 1
            if self.gripper_open_cnt > 10:
                # self.obj_in_the_hand = False
                pass
        
        # for rotate the valve
        # safe_gripper_target = 0.0

        self.safe_gripper_target = copy.deepcopy(safe_gripper_target)
        return safe_target_ee_pos, safe_target_ee_ori

        




    def start(self):
        while self.odom is None:
            rospy.logwarn_throttle(1, "No odometry received.")
            rospy.sleep(1)

        frame = self.odom.header.frame_id

        msg = PoseCtrlTarget()
        msg.header.frame_id = frame

        ee_tracking_target_msg = PoseCtrlTarget()
        
        
        arm_msg = JointState()
        arm_msg.position = [0.0, 0.0, 0.0, 0.0]
        arm_msg.name = ["motor0", "motor1", "motor2", "motor3", "gripper"]
        arm_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
        arm_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0]


        i = 0

        while not self.takeoff and not rospy.is_shutdown():
            rospy.loginfo_throttle(1, "Waiting for takeoff command.")
            rospy.sleep(1)

        rospy.loginfo("Publishing trajectory...")

        cur_base_pose, cur_base_quat = self.get_base_pose()
        base_pos = cur_base_pose
        base_quat = cur_base_quat
        joint = np.zeros(3)
        base_vel = np.zeros(3)
        thrust = np.zeros(3)
        
        self.traj = list(self.traj)

        t_index = 0.0

        base_pos_ub = self.base_pos_ub
        base_pos_lb = self.base_pos_lb

        self.ik_planner.base_constrain_callback(base_pos_ub, base_pos_lb)


        while not rospy.is_shutdown():

            tick = time.time()
            # print("time: ", tick)
            if t_index >= self.takeoff_duration and self.state_status == TAKEOFF:
                self.state_status = FREEFLIGHT

            t_index += self.dt

            # Decide whether to use MPC or IK path
            mpc_is_newer = False
            if self.state_status == FREEFLIGHT and self.horizon_buf is not None:
                if self._ik_target_time is None and self._horizon_time is not None:
                    mpc_is_newer = True
                elif self._ik_target_time is not None and self._horizon_time is not None:
                    mpc_is_newer = (self._horizon_time > self._ik_target_time)

            if mpc_is_newer and hasattr(self, 'planner'):
                # Use MPC to follow provided EE horizon (no IK safety filter here)
                current_pos = np.array([
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                ])
                current_vel = np.array([
                    self.odom.twist.twist.linear.x,
                    self.odom.twist.twist.linear.y,
                    self.odom.twist.twist.linear.z,
                ])

                # planner base orientation not used; keep identity downstream like IK sender
                current_euler = np.array([0.0, 0.0, 0.0])

                # Reference horizon from buffer
                p_ref = self.horizon_buf[:, :3]
                quat_ref_xyzw = self.horizon_buf[:, 3:7]
                quat_ref_wxyz = xyzw_to_wxyz(quat_ref_xyzw)

                # Prior input seed for solver
                u_prev = np.zeros(6)
                u_prev[3:6] = self.arm_joints

                force_cmd, p_mpc, v_mpc, arm_angle_opt, arm_angle_cmd = self.planner.optimize(
                    current_pos, current_vel, self.arm_joints, current_euler,
                    p_ref, quat_ref_wxyz, u_prev
                )

                pos_cmd = np.asarray(p_mpc)
                vel_cmd = np.asarray(v_mpc)
                thrust_cmd = np.asarray(force_cmd)

                # Optional disturbance observer
                if getattr(self, 'dob', None) is not None:
                    desired_acc = force_cmd / self.planner.mpc.acc2thrust_gain
                    dist_thrust = self.dob.update(current_vel, desired_acc)
                    if self.enable_dob:
                        thrust_cmd = thrust_cmd + dist_thrust

                # Arm command (raw gripper, no +7 offset for MPC branch)
                target_arm_deg = arm_angle_cmd * RAD_TO_DEG
                roll_joint_deg = np.array([self.rolldof_target * RAD_TO_DEG])
                arm_cmd = np.concatenate([target_arm_deg, roll_joint_deg])
                gripper_cmd = np.array([self.raw_gripper_target])

                # Publish base target (identity orientation to mirror IK sender)
                msg.header.stamp = rospy.Time.now()
                msg.pose.position = Point(*(pos_cmd))
                msg.pose.orientation.x = 0.0
                msg.pose.orientation.y = 0.0
                msg.pose.orientation.z = 0.0
                msg.pose.orientation.w = 1.0
                if self.send_vel:
                    msg.twist.linear = Point(*(vel_cmd))
                if self.send_thrust:
                    msg.thrust = Vector3(*(thrust_cmd))
                self.traj_pub.publish(msg)

                # Publish EE target from horizon for visibility
                ee_tracking_target_msg.header.stamp = rospy.Time.now()
                ee_tracking_target_msg.pose.position.x = float(p_ref[0, 0])
                ee_tracking_target_msg.pose.position.y = float(p_ref[0, 1])
                ee_tracking_target_msg.pose.position.z = float(p_ref[0, 2])
                ee_tracking_target_msg.pose.orientation.x = float(quat_ref_xyzw[0, 0])
                ee_tracking_target_msg.pose.orientation.y = float(quat_ref_xyzw[0, 1])
                ee_tracking_target_msg.pose.orientation.z = float(quat_ref_xyzw[0, 2])
                ee_tracking_target_msg.pose.orientation.w = float(quat_ref_xyzw[0, 3])
                self.ee_traj_pub.publish(ee_tracking_target_msg)

                # Publish arm command
                if self.send_arm:
                    cmd = np.concatenate([arm_cmd, gripper_cmd])
                    arm_msg.position = cmd.tolist()
                    arm_msg.header.stamp = rospy.Time.now()
                    self.arm_pub.publish(arm_msg)

                tock = time.time()
                self.IK_time.append(tock - tick)

                self.rate.sleep()
                continue

            # IK path (original behavior from IK sender)
            # Get the next ee waypoint
            new_waypoint = self.traj.pop(0)

            if len(self.traj) == 0:
                self.traj.append(new_waypoint)
                rospy.logwarn_once("Trajectory finished.")
            else:
                self.t_bar.update(1)

            ee_pos = new_waypoint[:3]
            ee_ori = new_waypoint[3:]

            for i in range(3):
                if ee_pos[i] > self.ee_pos_ub[i]:
                    ee_pos[i] = self.ee_pos_ub[i]
                elif ee_pos[i] < self.ee_pos_lb[i]:
                    ee_pos[i] = self.ee_pos_lb[i]

            safe_target_ee_pos, safe_target_ee_ori = self.ee_target_safe_filter(ee_pos, ee_ori)

            # Get the target base position and joint angles from the IK planner
            cur_base_pos, cur_base_quat = self.get_base_pose()

            target_base_pos, target_base_quat, target_arm_joints, real_target_ee_state = self.get_reference_state_fulldof(
                self.ik_planner,
                safe_target_ee_pos,
                safe_target_ee_ori,
                cur_base_pos,
                cur_base_quat,
                self.arm_joints,
                self.state_status,
            )

            if np.linalg.norm(target_arm_joints - self.last_arm_joint_angles) <= 40:
                self.last_arm_joint_angles = copy.deepcopy(target_arm_joints)
            else:
                target_arm_joints = copy.deepcopy(self.last_arm_joint_angles_cmd[:3])

            # Publish the whole-body trajectory (identity orientation)
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = Point(*(target_base_pos))
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0
            if self.send_vel:
                msg.twist.linear = Point(*(base_vel))
            if self.send_thrust:
                msg.thrust = Vector3(*(thrust))
            self.traj_pub.publish(msg)

            ee_tracking_target_msg.header.stamp = rospy.Time.now()
            ee_tracking_target_msg.pose.position.x = safe_target_ee_pos[0]
            ee_tracking_target_msg.pose.position.y = safe_target_ee_pos[1]
            ee_tracking_target_msg.pose.position.z = safe_target_ee_pos[2]
            ee_tracking_target_msg.pose.orientation.x = safe_target_ee_ori[0]
            ee_tracking_target_msg.pose.orientation.y = safe_target_ee_ori[1]
            ee_tracking_target_msg.pose.orientation.z = safe_target_ee_ori[2]
            ee_tracking_target_msg.pose.orientation.w = safe_target_ee_ori[3]
            self.ee_traj_pub.publish(ee_tracking_target_msg)

            real_ee_tracking_target_msg = PoseCtrlTarget()
            real_ee_tracking_target_msg.header.stamp = rospy.Time.now()
            real_ee_tracking_target_msg.pose.position = Point(*real_target_ee_state[:3])
            real_ee_tracking_target_msg.pose.orientation = Quaternion(*real_target_ee_state[3:])
            self.real_ee_tracking_target_pub.publish(real_ee_tracking_target_msg)

            # Publish arm command (IK branch keeps +7 offset and safe gripper)
            if self.send_arm:
                gripper_command = np.array([self.safe_gripper_target])
                roll_joint = np.array([self.rolldof_target]) * 180.0 / np.pi
                target_arm_joints[-1] += 7.0
                cmd = np.concatenate([target_arm_joints, roll_joint, gripper_command])
                self.last_arm_joint_angles_cmd = copy.deepcopy(cmd)
                arm_msg.position = cmd.tolist()
                arm_msg.header.stamp = rospy.Time.now()
                self.arm_pub.publish(arm_msg)
            

                    
            tock = time.time()
            self.IK_time.append(tock-tick)


            

            self.rate.sleep()


def main():
    nh = rospy.init_node("trajectory_planner")
    trajectory_planner = TrajectorySender()
    print("Trajectory planner initialized.")
    trajectory_planner.start()


if __name__ == "__main__":
    main()