#!/usr/bin/python3
# Xiaofeng @ 2025-01-16 0102
# What's this file for?

import numpy as np
from tqdm import tqdm
import numpy as np
import yaml

import rospy
from std_msgs.msg import Bool, Float64MultiArray
from geometry_msgs.msg import Quaternion, Point, Vector3

from nav_msgs.msg import Odometry
from core_pose_controller.msg import PoseCtrlTarget

from behavior_tree_msgs.msg import BehaviorTreeCommands
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def rpy_to_quaternion(roll, pitch, yaw):
    r = R.from_euler('xyz', [roll, pitch, yaw])
    quaternion = r.as_quat()
    return quaternion

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
        self.traj = trajectory
                        
        # traj: px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,gripper
        geofence = config.get("geofence", [5.0, 3.0, 2.0, -0.05, -3.0, 0.0])
        geofence_max = np.array(geofence[:3])
        geofence_min = np.array(geofence[3:])
        self.geofence = np.array([geofence_max, geofence_min])
        
        self.return_home_pos = config.get("return_home_pos", [0.0, 0.0, 1.25])
        self.return_home_quat = config.get("return_home_quat", [0.0, 0.0, 0.0, 1.0])
        self.return_home_pos = np.array(self.return_home_pos)
        self.return_home_quat = np.array(self.return_home_quat)
        
        # Check if the trajectory is a 16D vector
        for i in range(len(self.traj)):
            if self.traj[i].shape != (16,):
                print(self.traj[i].shape)
                raise ValueError(
                    f"{i}-th trajectory is not a 16D vector, expected 3D position, 4D quaterion, 3D velocity, 3D thrust, 2D joint angles, 1D gripper."
                )
            pos, quat, vel, thrust, joint, gripper = np.split(
                self.traj[i], [3, 7, 10, 13, 15]
            )
            if (pos < self.geofence[1]).any() or (pos > self.geofence[0]).any():
                raise ValueError(f"{i}-th traj waypoint is out of geofence.")

        self.t_bar = tqdm(total=len(self.traj))
        self.takeoff = False

        self.origin = config.get("origin", [0.0, 0.0, 0.0])
        self.takeoff_duration = config.get("takeoff_duration", 8.0)
        self.land_duration = config.get("land_duration", 8.0)
        
        
        self.traj_pub = rospy.Publisher(
            "/tracking_target", PoseCtrlTarget, queue_size=5
        )
        
        # self.arm_active_pub = rospy.Publisher("arm_active", Bool, queue_size=10)

        self.arm_pub = rospy.Publisher(
            "/manipulator_arm_command", Float64MultiArray, queue_size=10
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
                
        self.dt = config.get("dt", 0.01)
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


        self.origin = config.get("origin", [0.0, 0.0, 0.0])
        self.takeoff_duration = config.get("takeoff_duration", 8.0)
        self.land_duration = config.get("land_duration", 8.0)
        
        
        self.traj_pub = rospy.Publisher(
            "/tracking_target", PoseCtrlTarget, queue_size=5
        )
        self.ee_traj_pub = rospy.Publisher("/ee_tracking_target", PoseCtrlTarget, queue_size=5)

        self.arm_pub = rospy.Publisher(
            "/manipulator_arm_command", Float64MultiArray, queue_size=10
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

        self.arm_joint2_sub = rospy.Subscriber(
            "/arm_state", JointState, self.arm_state_callback, queue_size=10
        )


        # Subscribe to the online ee target position
        self.ee_target_sub = rospy.Subscriber("/hexa_scorpion/ee_target", PoseCtrlTarget, self.ee_target_callback, queue_size=10)


        




    def odom_callback(self, msg):
        self.odom = msg

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

        self.traj = self.takeoff_planner() + self.traj
        self.t_bar = tqdm(total=len(self.traj))

        self.takeoff = True
        rospy.loginfo("Starting takeoff sequence.")

    def land_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot land.")
            return

        self.traj = self.land_planner()
        self.t_bar = tqdm(total=len(self.traj))
        rospy.loginfo("Starting landing sequence.")

    def pause_callback(self, msg):
        if not self.takeoff:
            rospy.logerr("Not taking off, cannot pause.")
            return

        self.traj = [self.traj[0]]
        self.t_bar = tqdm(total=len(self.traj))
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
        self.t_bar = tqdm(total=len(self.traj))
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
        tar_pos = self.traj[0][:3]
        tar_quat = self.traj[0][3:7]
        start_pos = cur_pos.reshape((3, 1))
        end_pos = tar_pos.reshape((3, 1))
        pos_traj = self.cos_interp(start_pos, end_pos, self.takeoff_duration)
        start_quat = cur_quat.reshape((1, 4))
        end_quat = tar_quat.reshape((1, 4))
        quat_traj = self.slerp_interpolation(start_quat[0], end_quat[0], self.takeoff_duration)
        pos_quat_traj = np.hstack([pos_traj, quat_traj])
        state_dim = self.traj[0].shape[0]
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
        quat_traj = self.slerp_interpolation(start_quat[0], end_quat[0], self.takeoff_duration)
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

        frame = self.odom.header.frame_id

        msg = PoseCtrlTarget()
        msg.header.frame_id = frame
        
        arm_msg = Float64MultiArray()
        arm_msg.data = [0.0, 0.0, 0.0]

        i = 0

        while not self.takeoff and not rospy.is_shutdown():
            rospy.loginfo_throttle(1, "Waiting for takeoff command.")
            rospy.sleep(1)

        rospy.loginfo("Publishing trajectory...")

        while not rospy.is_shutdown():
            new_waypoint = self.traj.pop(0)
            pos, quat, vel, thrust, joint, gripper = np.split(
                new_waypoint, [3, 7, 10, 13, 15]
            )
            msg.header.stamp = rospy.Time.now()
            msg.pose.position = Point(*(pos))
            msg.pose.orientation = Quaternion(*(quat))
            if self.send_vel:
                msg.twist.linear = Point(*(vel))
            if self.send_thrust:
                msg.thrust = Vector3(*(thrust))
            self.traj_pub.publish(msg)

            # Publish arm command
            if self.send_arm:
                cmd = np.concatenate([joint, gripper])
                arm_msg.data = cmd.tolist()
                self.arm_pub.publish(arm_msg)
            
            # Check if the trajectory is finished, use the last waypoint as the final position
            if len(self.traj) == 0:
                new_waypoint = np.zeros_like(new_waypoint)
                new_waypoint[:3] = pos
                new_waypoint[3:7] = quat
                self.traj.append(new_waypoint)
                rospy.logwarn_once("Trajectory finished.")
            else:
                i += 1
                self.t_bar.update(1)

            self.rate.sleep()


def main():
    nh = rospy.init_node("trajectory_planner")
    trajectory_planner = TrajectorySender(list(traj))
    print("Trajectory planner initialized.")
    trajectory_planner.start()


if __name__ == "__main__":
    main()