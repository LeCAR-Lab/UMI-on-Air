#!/usr/bin/python3
import numpy as np
from tqdm import tqdm
import numpy as np
import yaml

import rospy
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Quaternion, Point, Vector3, Vector3Stamped, PoseStamped
from sensor_msgs.msg import JointState

from nav_msgs.msg import Odometry
from core_pose_controller.msg import PoseCtrlTarget, PoseCtrlInfo

from behavior_tree_msgs.msg import BehaviorTreeCommands
from planner.ee_mpc_acado_4dof import ArmMPCPlanner
from planner.ee_mpc_acado import DisturbanceObserver
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from planner.ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, quaternion_to_rpy

DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi

def xyzw_to_wxyz(quat):
    return np.concatenate((quat[:, 3:], quat[:, 0:3]), axis=1)



class TrajectorySender:
    def __init__(self, trajectory):
        config_path = rospy.get_param("~params", {})
        assert config_path, "No configuration found."
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        rospy.loginfo(f"Configuration loaded from: {config_path}")

        self.send_vel = config.get("send_vel", True)
        self.send_thrust = config.get("send_thrust", False)
        self.send_arm = config.get("send_arm", False)
        rospy.loginfo(f"Topic sending: vel: {self.send_vel}, thrust: {self.send_thrust}, arm: {self.send_arm}")
                
        self.dt = config.get("dt", 0.01)
        self.rate = rospy.Rate(1 / self.dt)
        self.odom = None
        self.mocap = None
        self.ee_traj = trajectory
                                
        # traj: px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,gripper
        geofence = config.get("geofence", [5.0, 3.0, 2.0, -0.05, -3.0, 0.0])
        self.geofence_max = np.array(geofence[:3])
        self.geofence_min = np.array(geofence[3:])
        self.geofence = np.array([self.geofence_max, self.geofence_min])
        
        self.takeoff_pos = np.array(config.get("takeoff_pos", [0.0, 0.0, 1.0]))
        self.takeoff_quat = np.array(config.get("takeoff_quat", [0.0, 0.0, 0.0, 1.0]))
        
        self.return_home_pos = np.array(config.get("return_home_pos", [0.0, 0.0, 1.25]))
        self.return_home_quat = np.array(config.get("return_home_quat", [0.0, 0.0, 0.0, 1.0]))
        self.return_home_pos = np.array(self.return_home_pos)
        self.return_home_quat = np.array(self.return_home_quat)
        
        # Check if the trajectory is a 17D vector
        self.traj = []
        for i in range(len(self.ee_traj)):
            if self.ee_traj[i].shape != (18,):
                print(self.ee_traj[i].shape)
                raise ValueError(
                    f"{i}-th trajectory is not a 17D vector, expected 3D position, 4D quaterion, 3D velocity, 3D thrust, 4D joint angles, 1D gripper."
                )
            pos, quat, vel, thrust, joint, gripper = np.split(
                self.ee_traj[i], [3, 7, 10, 13, 16]
            )
            if (pos < self.geofence[1]).any() or (pos > self.geofence[0]).any():
                raise ValueError(f"{i}-th traj waypoint is out of geofence, {pos}, {self.geofence}")

        self.t_bar = tqdm(total=len(self.ee_traj))
        self.takeoff = False
        self.use_mpc_cnt = -1

        self.origin = config.get("origin", [0.0, 0.0, 0.0])
        self.takeoff_duration = config.get("takeoff_duration", 8.0)
        self.land_duration = config.get("land_duration", 8.0)
        
        self.arm_joints = np.array([0.0, 0.0, 1.57])     # in radians
        
        
        self.traj_pub = rospy.Publisher(
            "/tracking_target", PoseCtrlTarget, queue_size=5
        )
        
        # self.arm_active_pub = rospy.Publisher("arm_active", Bool, queue_size=10)

        self.arm_pub = rospy.Publisher(
            "/manipulator_arm_command", JointState, queue_size=10
        )
        
        self.traj_ref_pub = rospy.Publisher(
            "/trajectory_reference", PoseCtrlTarget, queue_size=5
        )
        
        self.mpc_info_pub = rospy.Publisher(
            "/mpc_info", PoseCtrlInfo, queue_size=5
        )
        
        self.dob_pub = rospy.Publisher(
            "/disturbance_thrust", Vector3Stamped, queue_size=5
        )

        self.odom_sub = rospy.Subscriber(
            "/odometry", Odometry, self.odom_callback, queue_size=10
        )
        self.mocap_sub = rospy.Subscriber(
            "/mocap_node/hexascorpion/pose", PoseStamped, self.mocap_callback, queue_size=10
        )
        self.commander_sub = rospy.Subscriber(
            "/behavior_tree_commands",
            BehaviorTreeCommands,
            self.command_callback,
            queue_size=10,
        )
        self.arm_joint_sub = rospy.Subscriber(
            "/arm_state", JointState, self.arm_state_callback, queue_size=10
        )
        
        self.mpc_config = config["mpc"]
        mpc_params = self.mpc_config
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

        # Initialize MPC Planner with parameters
        self.planner = ArmMPCPlanner(mass, T, N, Q, R, R_arm_delta, acc2thrust_gain,
                                      pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max, default_arm_angle, output_filter_gain, moment_of_inertia)
           
        dob_params = config["disturbance_observer"]   
        self.dob = DisturbanceObserver(
            cutoff_freq=dob_params["cutoff_freq"],
            acc2thrust_gain=mpc_params["acc2thrust_gain"],
            dt=self.dt,
            acc_min=dob_params["acc_min"],
            acc_max=dob_params["acc_max"],
        )
        self.enable_dob = dob_params["enable"]
        
        self.arm_default_angle = np.array([30.0, 32.0, 90.0])
        
        

    def arm_state_callback(self, msg):
        current_arm_joint_angles = np.array(msg.position)
        self.arm_joints[0] = current_arm_joint_angles[0] * np.pi / 180.0
        self.arm_joints[1] = current_arm_joint_angles[1] * np.pi / 180.0
        self.arm_joints[2] = (current_arm_joint_angles[2] - 15.0) * np.pi / 180.0

    def odom_callback(self, msg):
        self.odom = msg
    
    def mocap_callback(self, msg):
        self.mocap = msg

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

        self.takeoff_traj = self.takeoff_planner()
        self.traj = self.takeoff_traj + self.traj
        self.t_bar = tqdm(total=len(self.traj) + len(self.ee_traj))

        self.takeoff = True
        self.use_mpc_cnt = len(self.takeoff_traj)
        rospy.loginfo("Starting takeoff sequence.")

    def land_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot land.")
            return

        self.traj = self.land_planner()
        self.t_bar = tqdm(total=len(self.traj))
        self.use_mpc_cnt = -1
        rospy.loginfo("Starting landing sequence.")

    def pause_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot pause.")
            return

        self.traj = [self.traj[0]]
        self.t_bar = tqdm(total=len(self.traj))
        self.use_mpc_cnt = -1
        rospy.loginfo("Pausing trajectory.")

    def returnhome_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot return home.")
            return

        cur_pose = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        cur_quat = np.array(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )
        tar_pos = self.return_home_pos
        tar_quat = self.return_home_quat
        start = cur_pose.reshape((3, 1))
        end = tar_pos.reshape((3, 1))
        pos_traj = self.line_interp(start, end, self.land_duration)
        start_quat = cur_quat.reshape((1, 4))
        end_quat = tar_quat.reshape((1, 4))
        quat_traj = self.slerp_interpolation(start_quat[0], end_quat[0], self.takeoff_duration)
        pos_quat_traj = np.hstack([pos_traj, quat_traj])
        state_dim = self.traj[0].shape[0]
        traj_dim = (len(pos_quat_traj), state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pos_quat_traj
        traj[:, 10:13] = self.arm_default_angle
        self.traj = list(traj)
        self.t_bar = tqdm(total=len(self.traj))
        self.use_mpc_cnt = -1
        rospy.loginfo("Returning home.")

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

    def takeoff_planner(self):
        if self.odom is None:
            rospy.logwarn("No odometry received.")
            return False

        cur_pos = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        cur_quat = np.array(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )
        tar_pos = self.takeoff_pos
        tar_quat = self.takeoff_quat
        start_pos = cur_pos.reshape((3, 1))
        end_pos = tar_pos.reshape((3, 1))
        pos_traj = self.cos_interp(start_pos, end_pos, self.takeoff_duration)
        start_quat = cur_quat.reshape((1, 4))
        end_quat = tar_quat.reshape((1, 4))
        quat_traj = self.slerp_interpolation(start_quat[0], end_quat[0], self.takeoff_duration)
        pos_quat_traj = np.hstack([pos_traj, quat_traj])
        state_dim = 17
        traj_dim = (len(pos_quat_traj), state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pos_quat_traj
        return list(traj)

    def land_planner(self):
        if self.odom is None:
            rospy.logwarn("No odometry received.")
            return False

        cur_pos = np.array(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ]
        )
        cur_quat = np.array(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )
        ground_z = self.origin[2]
        tar_pos = np.array([cur_pos[0], cur_pos[1], ground_z])
        tar_quat = np.array([0.0, 0.0, 0.0, 1.0])
        start = cur_pos.reshape((3, 1))
        end = tar_pos.reshape((3, 1))
        pos_traj = self.cos_interp(start, end, self.land_duration)
        start_quat = cur_quat.reshape((1, 4))
        end_quat = tar_quat.reshape((1, 4))
        quat_traj = self.slerp_interpolation(start_quat[0], end_quat[0], self.land_duration)
        pos_quat_traj = np.hstack([pos_traj, quat_traj])
        state_dim = self.traj[0].shape[0]
        traj_dim = (len(pos_quat_traj), state_dim)
        traj = np.zeros(traj_dim)
        traj[:, :7] = pos_quat_traj
        return list(traj)

    def start(self):
        while self.odom is None:
            rospy.logwarn_throttle(1, "No odometry received.")
            rospy.sleep(1)
            if rospy.is_shutdown():
                return

        frame = self.odom.header.frame_id

        msg = PoseCtrlTarget()
        msg.header.frame_id = frame
        
        ref_msg = PoseCtrlTarget()
        ref_msg.header.frame_id = frame
        
        arm_msg = JointState()
        arm_msg.position = [30.0, 32.0, 90.0, 0.0, 0.0]
        arm_msg.name = ["motor0", "motor1", "motor2", "motor3", "gripper"]
        arm_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0]
        arm_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        dob_msg = Vector3Stamped()
        dob_msg.header.frame_id = frame
        dob_msg.vector = Vector3()
        
        mpc_info_msg = PoseCtrlInfo()
        mpc_info_msg.header.frame_id = frame
        

        i = 0
        
        
        
        t0 = rospy.Time.now()
        t_pass = t0 + rospy.Duration(3.0)
        odom_thresh = 5e-3 # 1mm
        while rospy.Time.now() < t_pass:
            self.rate.sleep()
            rospy.loginfo_throttle(0.5, "Odometry & Mocap Sync Checking...")
            odom_pos = np.array(
                [
                    self.odom.pose.pose.position.x,
                    self.odom.pose.pose.position.y,
                    self.odom.pose.pose.position.z,
                ]
            )
            mocap_pos = np.array(
                [
                    self.mocap.pose.position.x,
                    self.mocap.pose.position.y,
                    self.mocap.pose.position.z,
                ]
            )
            
            if np.linalg.norm(odom_pos - mocap_pos) > odom_thresh:
                raise ValueError(f"Odometry & Mocap Sync Error: Odom: {odom_pos}, Mocap: {mocap_pos}, Error: {np.linalg.norm(odom_pos - mocap_pos)}")
        
        rospy.loginfo("Odometry & Mocap Synced.")
        
        

        while not self.takeoff and not rospy.is_shutdown():
            rospy.loginfo_throttle(1, "Waiting for takeoff command.")
            rospy.sleep(1)
        

        rospy.loginfo("Publishing trajectory...")

        u_prev = np.zeros(6)
        
        arm_angle_error = np.zeros(3)
        arm_angle_integral = np.zeros(3)
        arm_angle_tar = self.arm_joints * RAD_TO_DEG
        arm_angle_cur =  np.zeros(3)
        
        while not rospy.is_shutdown():

            current_pos = np.array([self.odom.pose.pose.position.x,
                                    self.odom.pose.pose.position.y,
                                    self.odom.pose.pose.position.z])
            current_vel = np.array([self.odom.twist.twist.linear.x,
                                    self.odom.twist.twist.linear.y,
                                    self.odom.twist.twist.linear.z])

            current_quat = np.array([self.odom.pose.pose.orientation.w,
                                    self.odom.pose.pose.orientation.x,
                                    self.odom.pose.pose.orientation.y,
                                    self.odom.pose.pose.orientation.z])
            current_euler = quaternion_to_rpy(current_quat)
            current_euler = np.array([0.0, 0.0, 0.0])
            
            current_ee_state = self.planner.forward_kinematics(current_pos, current_euler, self.arm_joints)
            
            current_ee_pos = current_ee_state[:3]
            current_ee_quat = current_ee_state[3:7]
            print("use_mpc_cnt: ", self.use_mpc_cnt)
            print("Current EE Pos: ", current_ee_pos)
            
            ee_pos_ref = None
            ee_quat_ref = None
            
            arm_angle_cur = self.arm_joints * RAD_TO_DEG
            # I_arm = np.array([15.0, 20.0, 20.0])
            I_arm = np.array([0.0, 0.0, 0.0])
            
            u_prev[3:6] = self.arm_joints
            
            if self.use_mpc_cnt == 0:
                
                ref_traj = self.ee_traj[: self.planner.mpc.N]
                
                if len(ref_traj) < self.planner.mpc.N:
                    ref_traj += [ref_traj[-1]] * (self.planner.mpc.N - len(ref_traj))
                ref_traj = np.array(ref_traj)
                
                p_ref = ref_traj[:, :3]
                ee_pos_ref = p_ref[0]
                quat_ref = ref_traj[:, 3:7]
                ee_quat_ref = quat_ref[0]
                quat_ref = xyzw_to_wxyz(quat_ref)
                
                force_cmd, p_mpc, v_mpc, arm_angle_opt, arm_angle_cmd = self.planner.optimize(current_pos, current_vel, self.arm_joints, current_euler, p_ref, quat_ref, u_prev)
                self.ee_traj.pop(0)
                if len(self.ee_traj) < self.planner.mpc.N:
                    rospy.logwarn("Trajectory length is less than MPC horizon, stopping MPC.")
                    self.use_mpc_cnt = -1
                

                if (p_mpc < self.geofence_min).any() or (p_mpc > self.geofence_max).any():
                    rospy.logwarn_throttle(1, f"Reached geofence, clipped position: {p_mpc}")

                p_mpc = np.clip(p_mpc, self.geofence_min, self.geofence_max)
                
                
                pos = p_mpc
                vel = v_mpc
                thrust = force_cmd
                
                
                arm_angle_tar = arm_angle_cmd * RAD_TO_DEG
                arm_angle_error = (arm_angle_tar - arm_angle_cur)
                arm_angle_integral += arm_angle_error * self.dt
                print(f"Arm tar: {arm_angle_tar}, Arm cur: {arm_angle_cur}, Arm error: {arm_angle_error}, Arm integral: {arm_angle_integral}")


                
                arm_angle = arm_angle_tar
                
                
                desired_acc = force_cmd / self.planner.mpc.acc2thrust_gain
                dist_thrust = self.dob.update(current_vel, desired_acc) #clipped
                
                dob_msg.vector = Vector3(*(dist_thrust))
                dob_msg.header.stamp = rospy.Time.now()
                self.dob_pub.publish(dob_msg)
                
                if self.enable_dob:
                    thrust += dist_thrust
                
                # joint = arm_angle.append(np.array([0.0]))
                joint = np.array([arm_angle[0], arm_angle[1], arm_angle[2], 0.0])
                gripper = np.array([0.0])
                
                quat = quat_ref[0]
                self.traj = [np.concatenate([pos, quat, vel, thrust, joint, gripper])]
                
            else:
                new_waypoint = self.traj.pop(0)
                pos, quat, vel, thrust, joint, gripper = np.split(new_waypoint, [3, 7, 10, 13, 16])
                joint = np.array([self.arm_default_angle[0], self.arm_default_angle[1], self.arm_default_angle[2], 0.0]) 
                gripper = np.array([0.0])
            
            joint[:3] = np.clip(joint[:3], self.planner.mpc.joint_min*RAD_TO_DEG, self.planner.mpc.joint_max*RAD_TO_DEG)
            thrust_min = self.planner.mpc.acc2thrust_gain * self.planner.mpc.acc_min
            thrust_max = self.planner.mpc.acc2thrust_gain * self.planner.mpc.acc_max
            thrust = np.clip(thrust, thrust_min, thrust_max)
            vel = np.clip(vel, self.planner.mpc.vel_min, self.planner.mpc.vel_max)

            msg.header.stamp = rospy.Time.now()
            msg.pose.position = Point(*(pos))
            msg.pose.orientation = Quaternion(*(quat))
            if self.send_vel:
                msg.twist.linear = Point(*(vel))
            if self.send_thrust:
                msg.thrust = Vector3(*(thrust))
            self.traj_pub.publish(msg)
            

            if ee_pos_ref is not None:
                ref_msg.header.stamp = rospy.Time.now()
                ref_msg.pose.position = Point(*(ee_pos_ref))
                ref_msg.pose.orientation = Quaternion(*(ee_quat_ref))
                ref_msg.twist.linear = Point()
                ref_msg.thrust = Vector3()
                self.traj_ref_pub.publish(ref_msg)

            
                ee_pos_err = current_ee_pos - ee_pos_ref
                mpc_info_msg.header.stamp = rospy.Time.now()
                mpc_info_msg.pos_error = Vector3(*(ee_pos_err))
                mpc_info_msg.pos_target = Vector3(*(ee_pos_ref))
                mpc_info_msg.pos_actual = Vector3(*(current_ee_pos))
                self.mpc_info_pub.publish(mpc_info_msg)
                np.set_printoptions(precision=3, suppress=True)
                print(f"Pos Error: {ee_pos_err}, Pos Ref: {ee_pos_ref}, Pos Actual: {current_ee_pos}")
            

            # Publish arm command
            if self.send_arm:
                cmd = np.concatenate([joint, gripper])
                arm_msg.position = cmd.tolist()
                self.arm_pub.publish(arm_msg)
                print(f"Arm Command: {cmd}, Arm Angle: {self.arm_joints*RAD_TO_DEG}")
            
            # Check if the trajectory is finished, use the last waypoint as the final position
            if len(self.traj) == 0:
                # new_waypoint = np.zeros_like(new_waypoint)
                # new_waypoint[:3] = ee_pos_ref
                # new_waypoint[3:7] = ee_quat_ref
                self.traj.append(new_waypoint)
                rospy.logwarn_once("Trajectory finished.")
            else:
                i += 1
                self.t_bar.update(1)

            if self.use_mpc_cnt > 0:
                self.use_mpc_cnt -= 1
            
            
            self.rate.sleep()


def main():
    nh = rospy.init_node("trajectory_planner")
    filename = rospy.get_param("~trajectory_file")
    traj = np.loadtxt(filename, delimiter=",", skiprows=1)
    print("Trajectory file loaded from: ", filename)
    print("Trajectory shape: ", traj.shape)
    traj = traj[:, 1:20]
    trajectory_planner = TrajectorySender(list(traj))
    print("Trajectory planner initialized.")
    trajectory_planner.start()


if __name__ == "__main__":
    main()