import yaml
import csv
import numpy as np
import casadi as cs
import tqdm
import time
import matplotlib.pyplot as plt
from ee_mpc_acado_4dof import ArmMPCPlanner
from ee_mpc_acado import DisturbanceObserver
from scipy.spatial.transform import Rotation as R
from planner.ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca

RAD_TO_DEG = 180.0 / np.pi

def wxyz_to_xyzw(quat):
    return np.concatenate((quat[:, 1:], quat[:, 0:1]), axis=1)

def xyzw_to_wxyz(quat):
    return np.concatenate((quat[:, 3:], quat[:, 0:3]), axis=1)


def quaternion_to_rpy(quat, xyzw: bool = False, degrees: bool = False):
    """Convert quaternion to roll-pitch-yaw (XYZ intrinsic) angles.

    Parameters
    ----------
    quat : array-like
        Quaternion in either w,x,y,z (default) or x,y,z,w (set ``xyzw=True``) order.
    xyzw : bool, optional
        Set ``True`` if the input is already (x, y, z, w).  Default assumes (w, x, y, z).
    degrees : bool, optional
        If ``True`` return degrees, otherwise radians.
    """
    if xyzw:
        quat_xyzw = quat
    else:
        # reorder w,x,y,z -> x,y,z,w for scipy
        qw, qx, qy, qz = quat
        quat_xyzw = [qx, qy, qz, qw]

    r = R.from_quat(quat_xyzw)
    # Use the same XYZ convention as the trajectory visualiser (roll-pitch-yaw)
    euler = r.as_euler('xyz', degrees=degrees)
    return euler

def rpy_to_quaternion(rpy, xyzw: bool = False, degrees: bool = False):
    """Convert roll-pitch-yaw (XYZ intrinsic) to quaternion.

    Parameters
    ----------
    rpy : array-like
        (roll, pitch, yaw) angles.
    xyzw : bool, optional
        If ``True`` return (x, y, z, w); otherwise return (w, x, y, z).
    degrees : bool, optional
        Whether the provided angles are in degrees.
    """
    r = R.from_euler('xyz', rpy, degrees=degrees)
    qx, qy, qz, qw = r.as_quat()
    if xyzw:
        return [qx, qy, qz, qw]
    else:
        return [qw, qx, qy, qz]



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
        self.sim_acc2thrust_gain = self.test_config['simulation']['acc2thrust_gain']
        self.sim_thrust_bias = self.test_config['simulation']['thrust_bias']
        self.trajectory_csv = self.test_config['trajectory']['csv_path']

        # Extract MPC parameters from mpc YAML
        mpc_params = self.mpc_config['mpc']
        mass = mpc_params['mass']
        T = mpc_params['T']
        N = mpc_params['N_mini']
        self.dt = T/N
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
        joint_vel_min = mpc_params['joint_vel_min']
        joint_vel_max = mpc_params['joint_vel_max']
        default_arm_angle = mpc_params['default_arm_angle']
        output_filter_gain = mpc_params['output_filter_gain']
        moment_of_inertia = mpc_params['moment_of_inertia']

        # Extract horizon and mini-horizon sizes
        N_horizon = mpc_params['N']          # full horizon steps
        N_mini = mpc_params.get('N_mini', N_horizon)  # fall back to full if not provided

        # Initialize MPC Planner with parameters (signature: mass, T, N, N_mini, ...)
        self.planner = ArmMPCPlanner(mass, T, N_mini, N_mini, Q, R, R_arm_delta, acc2thrust_gain,
                                     pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max,
                                     default_arm_angle, output_filter_gain, moment_of_inertia)

        # Initialize Disturbance Observer if enabled in the MPC config
        do_config = self.mpc_config.get('disturbance_observer', {})
        self.observer = None
        if do_config.get('enable', False):
            cutoff_freq = do_config['cutoff_freq']
            do_acc_min = do_config['acc_min']
            do_acc_max = do_config['acc_max']
            self.observer = DisturbanceObserver(cutoff_freq, np.array(acc2thrust_gain), 
                                                self.dt, do_acc_min, do_acc_max)

        # Load reference trajectory from CSV
        self.ref_traj = self.load_reference_trajectory(self.trajectory_csv)
        self.sim_time = len(self.ref_traj) * self.dt
        
        # Now using direct transformation approach (DH parameters and related variables removed)
        self.arm_angle_alpha = 1.0
        self.base_yaw_alpha = 1.0


    def forward_kinematics(self, base_pos, base_euler, arm_angle):
        '''
            input: base_pos, base_euler(rpy), arm_angle (4 DOF)
            output: ee_state: [ee_pos, ee_quat]
        '''
        # Use direct transformation chain matching MuJoCo XML structure (same as MPC implementation)
        joint_angles = arm_angle  # [joint1, joint2, joint3, joint4]

        # Base rotation matrix
        base_rotation_matrix = rpy_to_rotation_matrix(base_euler)
        
        # Build transformation chain exactly as in MuJoCo XML:
        # 1. base_link to arm_base_link: pos="0.088 0 0.0"
        pos_arm_base = np.array([0.088, 0.0, 0.0])
        
        # 2. arm_base_link to manipulation_link1_pitch_link: pos="0.0 0.0 0.06475"
        pos_link1 = pos_arm_base + np.array([0.0, 0.0, 0.06475])
        
        # 3. Apply joint1 rotation (Y-axis, pitch): axis="0 1 0"
        c1, s1 = np.cos(joint_angles[0]), np.sin(joint_angles[0])
        rot_joint1 = np.array([[c1, 0, s1],
                               [0, 1, 0],
                               [-s1, 0, c1]])
        
        # 4. manipulation_link1 to manipulation_link2: pos="-0.3795 0.0 0.059" quat="0.707 0.707 0 0"
        # The quat="0.707 0.707 0 0" represents a 90° rotation around X-axis
        pos_link2_local = np.array([-0.3795, 0.0, 0.059])
        pos_link2 = pos_link1 + np.dot(rot_joint1, pos_link2_local)
        
        # Link2 has initial rotation quat="0.707 0.707 0 0" = 90° around X
        rot_link2_initial = np.array([[1, 0, 0],
                                      [0, 0, -1],
                                      [0, 1, 0]])
        
        # 5. Apply joint2 rotation (Z-axis, yaw): axis="0 0 1"  
        c2, s2 = np.cos(joint_angles[1]), np.sin(joint_angles[1])
        rot_joint2 = np.array([[c2, -s2, 0],
                               [s2, c2, 0],
                               [0, 0, 1]])
        
        # Combined rotation for link2
        rot_link2 = np.dot(np.dot(rot_joint1, rot_link2_initial), rot_joint2)
        
        # 6. manipulation_link2 to manipulation_link3: pos="0.4475 0 0"
        pos_link3_local = np.array([0.4475, 0.0, 0.0])
        pos_link3 = pos_link2 + np.dot(rot_link2, pos_link3_local)
        
        # 7. Apply joint3 rotation (Z-axis, yaw): axis="0 0 1"
        c3, s3 = np.cos(joint_angles[2]), np.sin(joint_angles[2])
        rot_joint3 = np.array([[c3, -s3, 0],
                               [s3, c3, 0],
                               [0, 0, 1]])
        
        rot_link3 = np.dot(rot_link2, rot_joint3)
        
        # 8. manipulation_link3 to manipulation_link4: pos="0.071 0 0"
        pos_link4_local = np.array([0.071, 0.0, 0.0])
        pos_link4 = pos_link3 + np.dot(rot_link3, pos_link4_local)
        
        # 9. Apply joint4 rotation (negative X-axis, roll): axis="-1 0 0"
        c4, s4 = np.cos(joint_angles[3]), np.sin(joint_angles[3])
        rot_joint4 = np.array([[1, 0, 0],
                               [0, c4, s4],    # Note: positive because axis is "-1 0 0"
                               [0, -s4, c4]])
        
        rot_link4 = np.dot(rot_link3, rot_joint4)
        
        # 10. manipulation_link4 to ee: pos="0.01 0 0"
        # 11. ee to ee_tool: pos="0.14 0 0"
        # Total: 0.01 + 0.14 = 0.15
        pos_ee_local = np.array([0.15, 0.0, 0.0])
        pos_ee = pos_link4 + np.dot(rot_link4, pos_ee_local)
        
        # Apply base rotation and translation
        ee_pos_world = np.dot(base_rotation_matrix, pos_ee) + base_pos
        ee_rot_world = np.dot(base_rotation_matrix, rot_link4)
        
        # Convert rotation matrix to quaternion
        ee_quat = quaternion_from_rotation_matrix(ee_rot_world)
        
        return np.concatenate((ee_pos_world, ee_quat))




    def load_reference_trajectory(self, csv_path):
        # Assuming CSV has columns: t, px, py, pz, ...
        traj = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        ee_state_ref = traj[:, 1:8]
        return ee_state_ref

    def simulate_dynamics(self, p, v, base_euler, arm_angle, force_command, torque_cmd, arm_angle_ref, dt, mass):
        """
        Update position and velocity using simple Euler integration.
        Now includes proper torque-based roll, pitch, and yaw dynamics.
        """
        a = (force_command - self.sim_thrust_bias) * mass / self.sim_acc2thrust_gain          # acceleration = thrust / mass
        v_new = v + a * dt
        p_new = p + v_new * dt

        # Torque-based dynamics (simplified moment of inertia)
        I_x, I_y, I_z = 0.1, 0.1, 0.1  # Simplified moments of inertia
        
        # This is a simplification where torque is proportional to angular acceleration
        # And we directly integrate acceleration to get position. This matches the existing
        # simplified yaw dynamics in this test script.
        angular_acceleration = np.array([torque_cmd[0] / I_x, torque_cmd[1] / I_y, torque_cmd[2] / I_z])
        base_euler_new = base_euler + angular_acceleration * dt

        # First-order lag for arm angles
        arm_angle_new = arm_angle + (arm_angle_ref - arm_angle) * self.arm_angle_alpha * self.dt

        ee_state = self.forward_kinematics(p_new, base_euler_new, arm_angle_new)
        ee_pos = ee_state[:3]
        ee_quat = ee_state[3:7]
        ee_euler = quaternion_to_rpy(ee_quat)
        return p_new, v_new, base_euler_new, arm_angle_new, ee_pos, ee_euler

    def run(self):
        # Initialize simulation state: position and velocity
        ref0 = self.ref_traj[0]
        v = np.zeros(3)
        
        # Get reference end-effector position and orientation from first trajectory point
        ee_pos_ref0 = ref0[:3]
        ee_quat_ref0 = ref0[3:7]  # This is in xyzw format
        
        # Convert to wxyz format for internal calculations if needed
        ee_quat_ref0_wxyz = xyzw_to_wxyz(ee_quat_ref0.reshape(1, 4))[0]
        
        # Extract Euler angles from reference quaternion to match orientation better
        ee_euler_ref0 = quaternion_to_rpy(ee_quat_ref0_wxyz)
        
        # Set base yaw to match reference yaw - this helps align orientation
        base_yaw = ee_euler_ref0[2]  # Yaw component
        base_euler = np.array([0.0, 0.0, base_yaw])
        
        # Set arm configuration with 90-degree roll to match zero roll orientation
        arm_angle = np.array([0.0, 0.0, 0.0, 90.0])/180*np.pi
        
        # Adjust pitch if needed
        arm_angle[0] = ee_euler_ref0[1]  # Reference pitch
        
        # Initial position guess
        p = np.zeros(3)
        
        # Calculate initial end-effector pose with current configuration
        ee_state = self.forward_kinematics(p, base_euler, arm_angle)
        ee_pos_init = ee_state[:3]
        ee_quat_init = ee_state[3:7]
        
        # Calculate the offset between initial EE position and reference
        ee_offset = ee_pos_ref0 - ee_pos_init
        
        # Adjust base position to make EE position match the reference
        p = p + ee_offset
        
        # Verify the match after adjustment
        ee_state_final = self.forward_kinematics(p, base_euler, arm_angle)
        ee_pos_final = ee_state_final[:3]
        ee_quat_final = ee_state_final[3:7]
        
        # Convert quaternions to Euler angles for easier comparison
        ee_euler_init = quaternion_to_rpy(ee_quat_init)
        ee_euler_final = quaternion_to_rpy(ee_quat_final)
        
        print("Initial setup:")
        print(f"  Target position: {ee_pos_ref0}")
        print(f"  Final position: {ee_pos_final}")
        print(f"  Position error: {np.linalg.norm(ee_pos_final - ee_pos_ref0):.6f}")
        print(f"  Target orientation (euler): {ee_euler_ref0 * 180/np.pi}")
        print(f"  Final orientation (euler): {ee_euler_final * 180/np.pi}")
        print(f"  Arm angles (degrees): {arm_angle * 180/np.pi}")
        
        last_u = np.zeros(10)
        u_prev = np.zeros(10)
        u_prev[6:] = arm_angle

        # Setup logging for analysis
        time_steps = int(self.sim_time / self.dt)
        print(f"Running simulation for {time_steps} steps")
        history_p = []
        history_v = []
        history_ee_pos = []
        history_ee_euler = []
        history_arm_angle = []
        history_ee_pos_ref = []
        history_ee_euler_ref = []
        history_f = []
        history_arm_angle_ref = []
        history_costs = []  # Add cost history
        
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
            
            force_cmd, torque_cmd, p_opt, v_opt, arm_angle_opt, arm_angle_cmd, base_euler_opt, total_cost = self.planner.optimize(p, v, arm_angle, base_euler, ee_pos_ref, ee_quat_ref, u_prev)
            
            t_end = time.time()
            solve_times.append(t_end - t_start)

            # Use disturbance observer to adjust thrust command if enabled
            if self.observer is not None:
                dist_thrust = self.observer.update(v, (force_cmd/self.planner.mpc.mass))
                force_command = force_cmd + dist_thrust
            else:
                force_command = force_cmd

            # Update dynamics using the separate function
            p, v, base_euler, arm_angle, ee_pos, ee_euler = self.simulate_dynamics(p, v, base_euler, arm_angle, force_command, torque_cmd, arm_angle_cmd, self.dt, self.planner.mpc.mass)

            # Log data
            history_p.append(p.copy())
            history_v.append(v.copy())
            history_ee_pos_ref.append(ee_pos_ref[0].copy())  # current reference point
            ee_euler_ref = quaternion_to_rpy(ee_quat_ref[0])
            history_ee_euler_ref.append(ee_euler_ref)
            history_ee_pos.append(ee_pos.copy())
            history_ee_euler.append(ee_euler)
            history_arm_angle.append(arm_angle.copy())
            history_f.append(np.concatenate([force_command, torque_cmd]))
            history_arm_angle_ref.append(arm_angle_cmd.copy())
            history_costs.append(total_cost)  # Log the MPC cost

            # Update last command
            u_prev = np.concatenate((force_cmd, torque_cmd, arm_angle_cmd))

            # Update base_euler for next iteration
            base_euler = base_euler

        # Convert logs to numpy arrays
        history_p = np.array(history_p)
        history_v = np.array(history_v)
        history_ee_pos_ref = np.array(history_ee_pos_ref)
        history_ee_euler_ref = np.array(history_ee_euler_ref)
        history_ee_pos = np.array(history_ee_pos)
        history_ee_euler = np.array(history_ee_euler)
        history_arm_angle = np.array(history_arm_angle)
        history_f = np.array(history_f)
        history_arm_angle_ref = np.array(history_arm_angle_ref)
        history_costs = np.array(history_costs)  # Convert costs to numpy array


        np.set_printoptions(precision=3)
        print(f"Average solve time: {np.mean(solve_times)*1000} ms")
        print(f"Max solve time: {np.max(solve_times)*1000} ms")

        # Plotting results
        self.plot_results(history_p, history_ee_pos, history_ee_euler, history_ee_pos_ref, history_ee_euler_ref, history_f, history_arm_angle, history_arm_angle_ref, history_costs, results_folder="res")


    def plot_results(
        self,
        history_p,
        history_ee_pos,
        history_ee_euler,
        history_ee_pos_ref,
        history_ee_euler_ref,
        history_f,
        history_arm_angle,
        history_arm_angle_ref,
        history_costs,
        results_folder="res"
    ):
        import os
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        time_axis = np.linspace(0, self.sim_time, len(history_p))

        # Example constraints
        z_min, z_max = 0.0, 1.0
        p_min, p_max = self.planner.mpc.pos_min, self.planner.mpc.pos_max
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
        ee_euler_rmse = np.sqrt(np.mean((history_ee_euler - history_ee_euler_ref)**2, axis=0))
        print(f"EE Position RMSE: {ee_pos_rmse}")
        print(f"EE Euler RMSE: {ee_euler_rmse}")

        fig2, ax2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i, label in enumerate(['fx', 'fy', 'fz']):
            ax2[0].plot(time_axis, history_f[:, i], label=label)
        ax2[0].set_ylabel('Force (N)')
        ax2[0].set_title('Base Forces and Torques')
        ax2[0].legend()

        for i, label in enumerate(['tx', 'ty', 'tz']):
            ax2[1].plot(time_axis, history_f[:, i+3], label=label)
        ax2[1].set_ylabel('Torque (Nm)')
        ax2[1].set_xlabel('Time (s)')
        ax2[1].legend()

        fig2.tight_layout()
        fig2.savefig(os.path.join(results_folder, "controls.png"))
        plt.close(fig2)

        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, axis_name in enumerate(['Px', 'Py', 'Pz']):
            axs3[i].plot(time_axis, history_p[:, i], label=f'Base {axis_name}')
            axs3[i].axhline(y=p_min[i], color='k', linestyle='--', label='min')
            axs3[i].axhline(y=p_max[i], color='k', linestyle='--', label='max')
            axs3[i].set_ylabel(f'{axis_name} (m)')
            axs3[i].legend(loc='best')
        axs3[-1].set_xlabel('Time (s)')
        fig3.suptitle('Base Position vs Constraints')
        fig3.tight_layout()
        fig3.savefig(os.path.join(results_folder, "base_pos_constraints.png"))
        plt.close(fig3)

        fig4, axs4 = plt.subplots(4, 1, figsize=(20, 10), sharex=True)
        joint_names = ['Joint1 (Pitch)', 'Joint2 (Yaw1)', 'Joint3 (Yaw2)', 'Joint4 (Roll)']
        for i in range(4):
            axs4[i].plot(time_axis, history_arm_angle[:, i]*RAD_TO_DEG, label=joint_names[i])
            if i < 3:
                axs4[i].plot(time_axis, self.planner.mpc.default_arm_angle[i]*np.ones_like(time_axis)*RAD_TO_DEG, '--', label='Default')
            else:
                # For roll joint, show default angle (should be 0)
                axs4[i].plot(time_axis, self.planner.mpc.default_arm_angle[i]*np.ones_like(time_axis)*RAD_TO_DEG, '--', label='Default')
            axs4[i].axhline(y=arm_min[i]*RAD_TO_DEG, color='k', linestyle='--', label='min')
            axs4[i].axhline(y=arm_max[i]*RAD_TO_DEG, color='k', linestyle='--', label='max')
            axs4[i].plot(time_axis, history_arm_angle_ref[:, i]*RAD_TO_DEG, '--', label='Ref')
            axs4[i].set_ylabel('Angle (deg)')
            axs4[i].legend(loc='best')
        axs4[-1].set_xlabel('Time (s)')
        
        fig4.suptitle('Arm Angles vs Constraints')
        fig4.tight_layout()
        fig4.savefig(os.path.join(results_folder, "arm_angles_constraints.png"))
        plt.close(fig4)
        
        fig5, axs5 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, axis in enumerate(['x', 'y', 'z']):
            axs5[i].plot(time_axis, history_p[:, i], label='Base ' + axis)
            axs5[i].plot(time_axis, history_ee_pos[:, i], label='EE ' + axis)
            # For each 100 steps, draw a connection between base and ee
            for j in range(0, len(time_axis), 100):
                axs5[i].plot([time_axis[j], time_axis[j]], [history_p[j, i], history_ee_pos[j, i]], 'k--', linewidth=0.5)
                
                
            if i == 2:
                # Arm base is at base_pos + [0.088, 0, 0.06475] from the transformation chain
                arm_base_z_offset = 0.06475
                ee_p_z_min = history_p[:, 2] + arm_base_z_offset + z_min
                axs5[i].plot(time_axis, ee_p_z_min, color='r', linestyle='--', label='z_min')
                axs5[i].legend(loc='best')
                
            axs5[i].set_ylabel(axis + ' (m)')
            axs5[i].legend(loc='best')
        axs5[-1].set_xlabel('Time (s)')
        fig5.suptitle('Base and End-Effector Positions Over Time')
        fig5.tight_layout()
        fig5.savefig(os.path.join(results_folder, "base_and_ee_positions.png"))
        plt.close(fig5)
        
        # Plot MPC costs over time (without pruning)
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.plot(time_axis, history_costs, 'b-', linewidth=2, label='MPC Horizon Cost')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Cost Value')
        ax6.set_title('MPC Cost Evolution Over Time')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # Add some statistics as text
        cost_stats_text = f'Mean: {np.mean(history_costs):.3f}\n'
        cost_stats_text += f'Std: {np.std(history_costs):.3f}\n'
        cost_stats_text += f'Min: {np.min(history_costs):.3f}\n'
        cost_stats_text += f'Max: {np.max(history_costs):.3f}'
        ax6.text(0.02, 0.98, cost_stats_text, transform=ax6.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig6.tight_layout()
        fig6.savefig(os.path.join(results_folder, "mpc_costs.png"))
        plt.close(fig6)
        
        print(f"MPC Cost Statistics:")
        print(f"  Mean: {np.mean(history_costs):.3f}")
        print(f"  Std:  {np.std(history_costs):.3f}")
        print(f"  Min:  {np.min(history_costs):.3f}")
        print(f"  Max:  {np.max(history_costs):.3f}")
        
        

if __name__ == "__main__":
    env = DroneSimEnv('ee_mpc_test.yaml')
    env.run()
