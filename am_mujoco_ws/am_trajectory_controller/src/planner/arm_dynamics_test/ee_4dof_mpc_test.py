import yaml
import csv
import numpy as np
import casadi as cs
import tqdm
import time
import matplotlib.pyplot as plt
from ee_4dof_mpc import ArmMPCPlanner
from scipy.spatial.transform import Rotation as R
from planner.ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca

RAD_TO_DEG = 180.0 / np.pi

def wxyz_to_xyzw(quat):
    return np.concatenate((quat[:, 1:], quat[:, 0:1]), axis=1)

def xyzw_to_wxyz(quat):
    return np.concatenate((quat[:, 3:], quat[:, 0:3]), axis=1)


def quaternion_to_rpy(quat, xyzw = False, degrees = False):
    if xyzw:
        quat_xyzw = quat
    if not xyzw:
        qw, qx, qy, qz = quat
        quat_xyzw = [qx, qy, qz, qw]
    r = R.from_quat(quat_xyzw)
    euler = r.as_euler('zyx', degrees=degrees)
    euler = euler[::-1]
    return euler

def rpy_to_quaternion(rpy, xyzw = False, degrees = False):
    ypr = rpy[::-1]
    r = R.from_euler('zyx', ypr, degrees=False)
    quat = r.as_quat()
    qx, qy, qz, qw = quat
    if xyzw:
        quat_xyzw = [qx, qy, qz, qw]
        return quat_xyzw
    else:
        quat_wxyz = [qw, qx, qy, qz]
        return quat_wxyz



class DroneSimEnv:
    def __init__(self, test_config_path):
        # Load test configuration
        with open(test_config_path, 'r') as f:
            self.test_config = yaml.safe_load(f)

        # Load MPC configuration from the specified YAML file
        mpc_config_path = self.test_config['mpc_config_path']
        with open(mpc_config_path, 'r') as f:
            self.mpc_config = yaml.safe_load(f)

        # Extract simulation parameters from test YAML
        self.dt = self.test_config['simulation']['dt']
        self.trajectory_csv = self.test_config['trajectory']['csv_path']

        # Extract MPC parameters from mpc YAML
        mpc_params = self.mpc_config['mpc']
        T = mpc_params['T']
        N = mpc_params['N']
        Q = mpc_params['Q']
        R = mpc_params['R']
        R_arm_delta = mpc_params['R_delta']
        joint_min = mpc_params['joint_min']
        joint_max = mpc_params['joint_max']
        default_arm_angle = mpc_params['default_arm_angle']
        output_filter_gain = mpc_params['output_filter_gain']

        # Initialize MPC Planner with parameters
        self.planner = ArmMPCPlanner(T, N, Q, R, R_arm_delta, joint_min, joint_max, default_arm_angle, output_filter_gain)

        # Load reference trajectory from CSV
        self.ref_traj = self.load_reference_trajectory(self.trajectory_csv)
        self.ref_traj = self.ref_traj[:3000]
        self.sim_time = len(self.ref_traj) * self.dt
        
        self.DH_params = np.array([[-np.pi/2, 0, 0, 0.0],
                            [-np.pi/2, 0.362895306, 0.0, -np.pi],
                            [-np.pi/2, 0.00711424939, 0.0496796518, 0.0],
                            [-np.pi/2, 0.441054359, 0.0, -np.pi],
                            [np.pi/2, 0.00980744858, 0.0762684723, 0.0],
                            [-np.pi/2, 0.0, 0.0, -np.pi/2]])
        self.arm_base_pos = np.array([0.0664397079, 0.0, -0.0171154472])
        self.ee_base_pos = np.array([0.149832002, -0.0189594673, -0.00617417526])
        
        self.arm_angle_alpha = 5.0


    def generate_dh_matrix(self, alpha, a, d, theta):

        transform_matrix = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                            [0, np.sin(alpha), np.cos(alpha), d],
                            [0, 0, 0, 1]])

        return transform_matrix



    def forward_kinematics(self, base_pos, base_euler, arm_angle):
        '''
            input: u: [base_pos, base_euler(rpy), joint_angles]
            output: ee_state: [ee_pos, ee_quat]
        '''

        joint_angles = np.array([0.0, arm_angle[0], 0.0, arm_angle[1], 0.0,  arm_angle[2]])


        T = np.eye(4)

        for i in range(6):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
            dh_matrix = self.generate_dh_matrix(alpha, a, d, theta)
            T = np.dot(T, dh_matrix)

        ee_pos = np.array([self.ee_base_pos[0], self.ee_base_pos[1], self.ee_base_pos[2], 1])
        ee_pos = np.dot(T, ee_pos)
        ee_pos = ee_pos[:3]/ee_pos[3]
        ee_orientation_matrix = T[:3, :3]
        

        base_rotation_matrix = rpy_to_rotation_matrix(base_euler)
        ee_pos = ee_pos + self.arm_base_pos
        ee_pos = np.dot(base_rotation_matrix, ee_pos[:3])
        ee_pos = ee_pos + base_pos
        ee_orientation_matrix = np.dot(base_rotation_matrix, ee_orientation_matrix)
        ee_quat = quaternion_from_rotation_matrix(ee_orientation_matrix)
        ee_state = np.concatenate((ee_pos, ee_quat))

        return ee_state




    def load_reference_trajectory(self, csv_path):
        # Assuming CSV has columns: t, px, py, pz, ...
        traj = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        ee_state_ref = traj[:, 1:8]
        return ee_state_ref

    def simulate_dynamics(self, arm_angle, arm_angle_ref, dt):
        """
        Update position and velocity using simple Euler integration.
        This function can be updated later to reflect different dynamics.
        """
        p_new = np.zeros(3)
        # arm_angle_new = arm_angle_ref
        arm_angle_new = arm_angle + (arm_angle_ref - arm_angle) * self.arm_angle_alpha * self.dt + 0.0 * np.array([5.0, 5.0, 5.0]) * np.pi/180
        ee_state = self.forward_kinematics(p_new, np.zeros(3), arm_angle_new)
        ee_pos = ee_state[:3]
        ee_quat = ee_state[3:7]
        ee_euler = quaternion_to_rpy(ee_quat)
        return arm_angle_new, ee_pos, ee_euler

    def run(self):
        # Initialize simulation state: position and velocity
        ref0 = self.ref_traj[0]
        arm_angle = np.array([10.0, 12.0, 90.0])/180*np.pi
        base_euler = np.zeros(3)
        u_prev = arm_angle

        # Setup logging for analysis
        time_steps = int(self.sim_time / self.dt)
        print(f"Running simulation for {time_steps} steps")
        history_ee_pos = []
        history_ee_euler = []
        history_arm_angle = []
        history_ee_pos_ref = []
        history_ee_euler_ref = []
        history_arm_angle_ref = []
        
        solve_times = []

        # Simulation Loop
        for step in tqdm.tqdm(range(time_steps)):
            # Determine the reference trajectory over the MPC horizon
            horizon = self.planner.mpc.N
            start_idx = min(step, len(self.ref_traj)-1)
            end_idx = start_idx + horizon
            if end_idx > len(self.ref_traj):
                pad = np.repeat(self.ref_traj[-1][np.newaxis, :], end_idx - len(self.ref_traj), axis=0)
                ee_state_refs = np.vstack((self.ref_traj[start_idx:], pad))
            else:
                ee_state_refs = self.ref_traj[start_idx:end_idx]
            
            

            # Optimize control using MPC
            ee_pos_ref = ee_state_refs[:, :3]
            ee_quat_ref = ee_state_refs[:, 3:7]
            ee_quat_ref = xyzw_to_wxyz(ee_quat_ref)

            
            t_start = time.time()
            
            
            
            arm_angle_opt, arm_angle_cmd = self.planner.optimize(arm_angle, base_euler, ee_pos_ref, ee_quat_ref, u_prev)
            # arm_angle_cmd = np.concatenate([self.planner.mpc.default_arm_angle, np.array([1.57])])
            
            t_end = time.time()
            solve_times.append(t_end - t_start)


            # Update dynamics using the separate function
            arm_angle, ee_pos, ee_euler = self.simulate_dynamics(arm_angle, arm_angle_cmd, self.dt)

            # Log data
            history_ee_pos_ref.append(ee_pos_ref[0].copy())  # current reference point
            ee_euler_ref = quaternion_to_rpy(ee_quat_ref[0])
            history_ee_euler_ref.append(ee_euler_ref)
            history_ee_pos.append(ee_pos.copy())
            history_ee_euler.append(ee_euler)
            history_arm_angle.append(arm_angle.copy())
            history_arm_angle_ref.append(arm_angle_cmd.copy())
            

            # Update last command
            # last_u = u_cmd
            u_prev = arm_angle_cmd

        # Convert logs to numpy arrays
        history_ee_pos_ref = np.array(history_ee_pos_ref)
        history_ee_euler_ref = np.array(history_ee_euler_ref)
        history_ee_pos = np.array(history_ee_pos)
        history_ee_euler = np.array(history_ee_euler)
        history_arm_angle = np.array(history_arm_angle)
        history_arm_angle_ref = np.array(history_arm_angle_ref)


        np.set_printoptions(precision=3)
        print(f"Average solve time: {np.mean(solve_times)*1000} ms")
        print(f"Max solve time: {np.max(solve_times)*1000} ms")

        # Plotting results
        self.plot_results(history_ee_pos, history_ee_euler, history_ee_pos_ref, history_ee_euler_ref, history_arm_angle, history_arm_angle, results_folder="res")


    def plot_results(
        self,
        history_ee_pos,
        history_ee_euler,
        history_ee_pos_ref,
        history_ee_euler_ref,
        history_arm_angle,
        history_arm_angle_ref,
        results_folder="res"
    ):
        import os
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        time_axis = np.linspace(0, self.sim_time, len(history_ee_pos))

        # Example constraints
        z_min, z_max = -0.1, 1.0
        arm_min, arm_max = self.planner.mpc.joint_min, self.planner.mpc.joint_max

        fig1, axs1 = plt.subplots(3, 2, figsize=(20, 8), sharex=True)
        for i, label in enumerate(['X_ee', 'Y_ee', 'Z_ee']):
            axs1[i, 0].plot(time_axis, history_ee_pos[:, i], label=f'Actual {label}')
            axs1[i, 0].plot(time_axis, history_ee_pos_ref[:, i], '--', label=f'Ref {label}')
            axs1[i, 0].set_ylabel(f'{label} (m)')
            axs1[i, 0].legend(loc='best')
            # if label == 'Z_ee':
            #     ee_p_z_min = history_p[:, 2] + self.arm_base_pos[2] + z_min
            #     ee_p_z_max = history_p[:, 2] + self.arm_base_pos[2] + z_max
            #     axs1[i].plot(time_axis, ee_p_z_min, color='r', linestyle='--', label='z_min')
            #     axs1[i].plot(time_axis, ee_p_z_max, color='r', linestyle='--', label='z_max')
            #     axs1[i].legend(loc='best')
        axs1[-1, 0].set_xlabel('Time (s)')
        
        for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
            axs1[i, 1].plot(time_axis, history_ee_euler[:, i] * RAD_TO_DEG, label=f'Actual {label}')
            axs1[i, 1].plot(time_axis, history_ee_euler_ref[:, i] * RAD_TO_DEG, '--', label=f'Ref {label}')
            axs1[i, 1].set_ylabel(f'{label} (deg)')
            axs1[i, 1].legend(loc='best')
        axs1[-1, 1].set_xlabel('Time (s)')
        
        fig1.suptitle('End-Effector Position Tracking & Constraints')
        fig1.tight_layout()
        fig1.savefig(os.path.join(results_folder, "ee_pos_tracking.png"))
        plt.close(fig1)
        
        ee_pos_rmse = np.sqrt(np.mean((history_ee_pos - history_ee_pos_ref)**2, axis=0))
        ee_pitch_rmse = np.sqrt(np.mean((history_ee_euler[:, 1] - history_ee_euler_ref[:, 1])**2)) * RAD_TO_DEG
        print(f"EE Position RMSE: {ee_pos_rmse}")
        print(f"EE Pitch RMSE: {ee_pitch_rmse}")


        fig4, axs4 = plt.subplots(3, 2, figsize=(20, 8), sharex=True)
        for i, axis_name in enumerate(['Joint1', 'Joint2', 'Joint3']):
            axs4[i, 0].plot(time_axis, history_arm_angle[:, i]*RAD_TO_DEG, label=axis_name)
            if i < 2:
                axs4[i, 0].plot(time_axis, self.planner.mpc.default_arm_angle[i]*np.ones_like(time_axis)*RAD_TO_DEG, '--', label='Default')
            axs4[i, 0].axhline(y=arm_min[i]*RAD_TO_DEG, color='k', linestyle='--', label='min')
            axs4[i, 0].axhline(y=arm_max[i]*RAD_TO_DEG, color='k', linestyle='--', label='max')
            axs4[i, 0].set_ylabel('Angle (deg)')
            axs4[i, 0].legend(loc='best')
        axs4[-1, 0].set_xlabel('Time (s)')
        
        for i, axis_name in enumerate(['Joint1', 'Joint2', 'Joint3']):
            axs4[i, 1].plot(time_axis, history_arm_angle_ref[:, i]*RAD_TO_DEG, label=axis_name)
            axs4[i, 1].axhline(y=self.planner.mpc.joint_min[i]*RAD_TO_DEG, color='k', linestyle='--', label='min')
            axs4[i, 1].axhline(y=self.planner.mpc.joint_max[i]*RAD_TO_DEG, color='k', linestyle='--', label='max')
            axs4[i, 1].set_ylabel('Angle Command (deg)')
            axs4[i, 1].legend(loc='best')
        axs4[-1, 1].set_xlabel('Time (s)')
        
        fig4.suptitle('Arm Angles vs Constraints')
        fig4.tight_layout()
        fig4.savefig(os.path.join(results_folder, "arm_angles_constraints.png"))
        plt.close(fig4)
        
        
        fig5, axs5 = plt.subplots(1, 1, figsize=(12, 6))
        axs5.plot(time_axis, history_ee_pos[:, 2], label='Z_ee')
        axs5.plot(time_axis, history_ee_pos_ref[:, 2], '--', label='Z_ee_ref')
        axs5.plot(time_axis, self.arm_base_pos[2] + z_min*np.ones_like(time_axis), color='r', linestyle='--', label='z_min')
        axs5.legend(loc='best')
        fig5.suptitle('End-Effector Z Position Tracking & Constraints')
        fig5.tight_layout()
        fig5.savefig(os.path.join(results_folder, "ee_z_tracking.png"))
        plt.close(fig5)
        
        
        
        

if __name__ == "__main__":
    env = DroneSimEnv('ee_4dof_test.yaml')
    env.run()
