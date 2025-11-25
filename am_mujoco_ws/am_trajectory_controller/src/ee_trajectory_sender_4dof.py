#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
import numpy as np
import yaml
import time
import logging
import copy
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

# Define the global states
from planner.ee_ik_DH_4dof import FREEFLIGHT, TAKEOFF, LAND, PAUSE


# TEST logging
TEST_FLAG = False
OUTPUT_FILE = "output.csv"
logger = logging.getLogger("IK_output")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(OUTPUT_FILE)
fh.setLevel(logging.INFO)
logger.addHandler(fh)
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)



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







class TrajectorySender:
    def __init__(self, trajectory):
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
                
        self.dt = config.get("dt", 0.05)
        self.rate = rospy.Rate(1 / self.dt)
        self.odom = None

        self.ee_traj = trajectory
        # print("ee traj shape", self.ee_traj.shape)
        # self.base_traj = np.zeros((len(self.ee_traj), 9))
        # self.len_traj = len(self.ee_traj)
                        
        # ee_traj: px,py,pz
        geofence = config.get("geofence", [5.0, 3.0, 2.0, -0.05, -3.0, 0.0])
        geofence_max = np.array(geofence[:3])
        geofence_min = np.array(geofence[3:])
        self.geofence = np.array([geofence_max, geofence_min])
        
        self.return_home_pos = config.get("return_home_pos", [0.0, 0.0, 1.25])
        self.return_home_quat = config.get("return_home_quat", [0.0, 0.0, 0.0, 1.0])
        self.return_home_pos = np.array(self.return_home_pos)
        self.return_home_quat = np.array(self.return_home_quat)
        
        # Check if the trajectory is a 7D vector
        for i in range(len(self.ee_traj)):
            if self.ee_traj[i].shape != (7,):
                print(self.ee_traj[i].shape)
                raise ValueError(
                    f"{i}-th trajectory is not a 7D vector, expected 3D position and 4D quaternion for the end-effector."
                )
            pos = self.ee_traj[i, :3]
            if (pos < self.geofence[1]).any() or (pos > self.geofence[0]).any():
                raise ValueError(f"{i}-th traj waypoint is out of geofence.")

        self.t_bar = tqdm(total=len(self.ee_traj))
        self.takeoff = False

        self.origin = config.get("origin", [0.0, 0.0, 0.0])
        self.takeoff_duration = config.get("takeoff_duration", 8.0)
        self.land_duration = config.get("land_duration", 8.0)
        
        self.traj_pub = rospy.Publisher(
            "/tracking_target", PoseCtrlTarget, queue_size=5
        )
        
        self.ee_traj_pub = rospy.Publisher("/ee_tracking_target", PoseCtrlTarget, queue_size=5)

        # publish the real final ee tracking point
        self.real_ee_tracking_target_pub = rospy.Publisher("/real_ee_tracking_target", PoseCtrlTarget, queue_size=5)


        # publish the current ee state
        self.ee_state_pub = rospy.Publisher("/ee_state", PoseCtrlTarget, queue_size=5)

        # self.arm_active_pub = rospy.Publisher("arm_active", Bool, queue_size=10)

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
        # self.arm_joint1_sub = rospy.Subscriber(
        #     "/hexa_scorpion/manipulator_arm1_joint/angle", Float32, self.arm_joint1_callback, queue_size=10
        # )
        # self.arm_joint2_sub = rospy.Subscriber(
        #     "/hexa_scorpion/manipulator_arm2_joint/angle", Float32, self.arm_joint2_callback, queue_size=10
        # )
        self.arm_joint_sub = rospy.Subscriber(
            "/arm_state", JointState, self.arm_state_callback, queue_size=10
        )
        
        self.gripper_target = 1.0
        self.rolldof_target = 0.0


        self.arm_joints = np.array([0.0, 0.0, 1.57])     # in radians

        # self.mujoco_ee_offset = np.array([0.18506, 0.0, 0.102672])
        # self.base_pos_initial = np.array([0.0, 0.0, 0.22])
        self.after_takeoff_ee_pos_target = np.array([0.0, 0.0, 1.1])
        self.takeoff_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # self.inital_ee_offset = np.array([0.18, 0.0, 0.1])
        # self.takeoff_base_pos = self.ee_traj[0, :3] - self.inital_ee_offset

        self.state_status = TAKEOFF

        # IK Planner
        # self.ik_planner = init_ik_Mj(xml_path,
        #                              cur_base_pos=self.base_pos_initial, 
        #                              cur_base_quat=np.array([0.0, 0.0, 0.0, 1.0]), 
        #                              cur_joints=np.array([0.0, 0.0]),
        #                              visualize=True)

        self.ik_planner = init_ik_DH()
        rospy.loginfo("IK planner initialized.")
        
        self.IK_time = []
        
        self.timer = rospy.Timer(rospy.Duration(1.0), self.solver_fps_timer)

        # self.land_arm_joint_angles = np.array([0.0, 0.0])
        # self.pause_arm_joint_angles = np.array([0.0, 0.0])
        # self.last_arm_joint_angles = np.array([0.0, 0.0])

        self.land_arm_joint_angles = np.array([10.0, 10.0, 100])       # in degrees
        self.pause_arm_joint_angles = np.array([10.0, 10.0, 90])      # in degrees 
        self.last_arm_joint_angles = np.array([10.0, 10.0, 90])       # in degrees
        self.last_arm_joint_angles_cmd = np.array([10.0, 10.0, 90])       # in degrees

        # end-effector position constraints
        self.ee_pos_ub = np.array([2.7, 2.0, 2.0])
        self.ee_pos_lb = np.array([-1.0, -2.0, 0.0])


        self.base_pos_ub = np.array([0.5, 2.0, 2.0])
        self.base_pos_lb = np.array([-1.0, -2.0, 0.0])

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

    # def arm_joint1_callback(self, msg):
    #     # self.arm_joints: radian
    #     self.arm_joints[0] = msg.data
    
    # def arm_joint2_callback(self, msg):
    #     # self.arm_joints: radian

    #     self.arm_joints[1] = msg.data

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

    # def pause_callback(self, msg):
    #     if not self.takeoff:
    #         rospy.logerr("Not taking off, cannot pause.")
    #         return

    #     self.traj = list(self.pause_planner())
    #     self.state_status = PAUSE
        
    #     self.t_bar = tqdm(total=len(self.traj))
    #     rospy.loginfo("Pausing trajectory.")

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
        state_dim = pose_traj[0].shape[0]
        traj_dim = (len(pose_traj), state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pose_traj
        self.traj = list(traj)
        self.t_bar = tqdm(total=len(self.traj))
        rospy.loginfo("Returning home.")
    
    # def pause_planner(self):
    #     if self.odom is None:
    #         rospy.logwarn("No odometry received.")
    #         return False

    #     start_base_pos = np.array(
    #         [
    #             self.odom.pose.pose.position.x,
    #             self.odom.pose.pose.position.y,
    #             self.odom.pose.pose.position.z,
    #         ]
    #     )
    #     start_base_quat = np.array(
    #         [
    #             self.odom.pose.pose.orientation.x,
    #             self.odom.pose.pose.orientation.y,
    #             self.odom.pose.pose.orientation.z,
    #             self.odom.pose.pose.orientation.w,
    #         ]
    #     )
    #     # Interpolate the position
    #     state_dim = self.ee_traj[0].shape[0]
    #     traj_dim = (1, state_dim)
    #     traj = np.zeros(traj_dim)
    #     traj[:, :3] = start_base_pos
    #     traj[:, 3:7] = start_base_quat

    #     self.pause_arm_joint_angles = copy.deepcopy(self.arm_joints)

    #     return traj

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
        quat_traj = self.slerp_interpolation(start_ee_quat, target_ee_quat, self.takeoff_duration)
        # Combine the position and quaternion trajectory
        pose_traj = np.hstack([pos_traj, quat_traj])
        
        state_dim = self.ee_traj[0].shape[0]
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
        
        state_dim = self.ee_traj[0].shape[0]
        traj_dim = (pose_traj.shape[0], state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pose_traj

        self.land_arm_joint_angles = copy.deepcopy(self.arm_joints)

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


            # # Update the base position, quaternion and joint angles
            # target_base_pos = km_target[:3]
            # # mujoco quat: w,x,y,z -> gazebo quat: x,y,z,w
            # target_base_quat = np.array([km_target[4], km_target[5], km_target[6], km_target[3]])
            # target_arm_joints = km_target[7:9] / np.pi * 180.0
            # print("base pos: ", base_pos)
            # print("base quat: ", base_quat)
            # print("joint angles: ", target_arm_joints)
        # print("current arm joints: ", arm_joints)
        # print("target arm joints: ", target_arm_joints)
        return target_base_pos, target_base_ori, target_arm_joints, real_target_ee_state


    def start(self):

        if TEST_FLAG:
            odom_msg = Odometry()
            odom_msg.pose.pose.position.x = self.base_pos_initial[0]
            odom_msg.pose.pose.position.y = self.base_pos_initial[1]
            odom_msg.pose.pose.position.z = self.base_pos_initial[2]
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = 0.0
            odom_msg.pose.pose.orientation.w = 1.0
            
            self.odom_callback(odom_msg)
            self.takeoff_callback(None)


        while self.odom is None and not rospy.is_shutdown():
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
        roll_joint = np.zeros(1)
        gripper = np.zeros(1)
        self.traj = list(self.traj)
        
        t_index = 0.0
        # t = 0
        
        while not rospy.is_shutdown():
            
            tick = time.time()
            
            if t_index >= self.takeoff_duration and self.state_status == TAKEOFF:
                self.state_status = FREEFLIGHT


            t_index += self.dt
            
            # Get the next ee waypoint
            new_waypoint = self.traj.pop(0)

            target_ee_pos = new_waypoint[:3]
            target_ee_quat = new_waypoint[3:]
            target_ee_quat = np.array([0.0, 0.0, 0.0, 1.0])

            for i in range(3):
                if target_ee_pos[i] > self.ee_pos_ub[i]:
                    target_ee_pos[i] = self.ee_pos_ub[i]
                elif target_ee_pos[i] < self.ee_pos_lb[i]:
                    target_ee_pos[i] = self.ee_pos_lb[i]


            # Get the target base position and joint angles from the IK planner
            current_base_pos, current_base_quat = self.get_base_pose()
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



            # Publish the whole-body trajectory
            msg.header.stamp = rospy.Time.now()
            # print("target base pos: ", base_pos)
            msg.pose.position = Point(*(target_base_pos))
            # msg.pose.orientation = Quaternion(*(target_base_quat))
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
            ee_tracking_target_msg.pose.position = Point(*(target_ee_pos))
            ee_tracking_target_msg.pose.orientation.x = target_ee_quat[0]
            ee_tracking_target_msg.pose.orientation.y = target_ee_quat[1]
            ee_tracking_target_msg.pose.orientation.z = target_ee_quat[2]
            ee_tracking_target_msg.pose.orientation.w = target_ee_quat[3]
            self.ee_traj_pub.publish(ee_tracking_target_msg)

            real_ee_tracking_target_msg = PoseCtrlTarget()
            real_ee_tracking_target_msg.header.stamp = rospy.Time.now()
            real_ee_tracking_target_msg.pose.position = Point(*real_target_ee_state[:3])
            # real_ee_tracking_target_msg.pose.orientation = Quaternion(*real_target_ee_state[3:])
            real_ee_tracking_target_msg.pose.orientation.x = real_target_ee_state[3]
            real_ee_tracking_target_msg.pose.orientation.y = real_target_ee_state[4]
            real_ee_tracking_target_msg.pose.orientation.z = real_target_ee_state[5]
            real_ee_tracking_target_msg.pose.orientation.w = real_target_ee_state[6]
            self.real_ee_tracking_target_pub.publish(real_ee_tracking_target_msg)


            # Publish arm command
            if self.send_arm:

                gripper_command = np.array([self.gripper_target])
                roll_joint = np.array([self.rolldof_target])*180.0/np.pi
                target_arm_joints[-1] += 7.0
                cmd = np.concatenate([target_arm_joints, roll_joint, gripper_command])
                self.last_arm_joint_angles_cmd = copy.deepcopy(cmd)
                # arm_msg.data = cmd.tolist()
                arm_msg.position = cmd.tolist()
                arm_msg.header.stamp = rospy.Time.now()
                self.arm_pub.publish(arm_msg)
            
            
            tock = time.time()
            self.IK_time.append(tock-tick)
            
            # csv logging
            if TEST_FLAG:
                logger.info(
                    f"{t_index},{base_pos[0]},{base_pos[1]},{base_pos[2]},{base_quat[0]},{base_quat[1]},{base_quat[2]},{base_quat[3]},{joint[0]},{joint[1]}"
                )
            
            
            # Check if the trajectory is finished, use the last waypoint as the final position
            if len(self.traj) == 0:
                # new_waypoint = np.zeros_like(new_waypoint)
                # new_waypoint[:3] = base_pos
                # new_waypoint[3:7] = base_quat
                self.traj.append(new_waypoint)
                rospy.logwarn_once("Trajectory finished.")
                if TEST_FLAG:
                    break
            else:
                # t += 1
                self.t_bar.update(1)
                
            
            self.rate.sleep()

def main():
    
    if TEST_FLAG:
        rospy.logerr("!!!!! Running in test mode, press Enter to start the trajectory !!!!!")
        input()
    
    nh = rospy.init_node("trajectory_planner")
    filename = rospy.get_param("~trajectory_file")
    traj = np.loadtxt(filename, delimiter=",", skiprows=1)
    print("Trajectory file loaded from: ", filename)
    traj = traj[:, 1:8]
    trajectory_planner = TrajectorySender(traj)
    print("Trajectory planner initialized.")
    trajectory_planner.start()

if __name__ == "__main__":
    main()