import yaml
import csv
import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from ee_mpc_acado import BaseMPCPlanner, DisturbanceObserver

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
        self.sim_acc2thrust_gain = self.test_config['simulation']['acc2thrust_gain']
        self.sim_thrust_bias = self.test_config['simulation']['thrust_bias']
        self.trajectory_csv = self.test_config['trajectory']['csv_path']

        # Extract MPC parameters from mpc YAML
        mpc_params = self.mpc_config['mpc']
        mass = mpc_params['mass']
        T = mpc_params['T']
        N = mpc_params['N']
        Q = mpc_params['Q']
        R = mpc_params['R']
        acc2thrust_gain = mpc_params['acc2thrust_gain']
        pos_min = mpc_params['pos_min']
        pos_max = mpc_params['pos_max']
        vel_min = mpc_params['vel_min']
        vel_max = mpc_params['vel_max']
        acc_min = mpc_params['acc_min']
        acc_max = mpc_params['acc_max']

        # Initialize MPC Planner with parameters
        self.planner = BaseMPCPlanner(mass, T, N, Q, R, acc2thrust_gain,
                                      pos_min, pos_max, vel_min, vel_max, acc_min, acc_max)

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


    def load_reference_trajectory(self, csv_path):
        # Assuming CSV has columns: t, px, py, pz, ...
        traj = []
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Only consider positions px, py, pz as reference
                traj.append([float(row['px']), float(row['py']), float(row['pz'])])
        return np.array(traj)

    def simulate_dynamics(self, p, v, thrust_command, dt, mass):
        """
        Update position and velocity using simple Euler integration.
        This function can be updated later to reflect different dynamics.
        """
        a = (thrust_command - self.sim_thrust_bias) / self.sim_acc2thrust_gain          # acceleration = thrust / mass
        v_new = v + a * dt
        p_new = p + v_new * dt
        return p_new, v_new

    def run(self):
        # Initialize simulation state: position and velocity
        p = np.zeros(3)
        v = np.zeros(3)
        last_u = np.zeros(3)

        # Setup logging for analysis
        time_steps = int(self.sim_time / self.dt)
        print(f"Running simulation for {time_steps} steps")
        history_p = []
        history_v = []
        history_ref = []
        history_u = []

        # Simulation Loop
        for step in range(time_steps):
            # Determine the reference trajectory over the MPC horizon
            horizon = self.planner.mpc.N
            start_idx = min(step, len(self.ref_traj)-1)
            end_idx = start_idx + horizon
            if end_idx > len(self.ref_traj):
                pad = np.repeat(self.ref_traj[-1][np.newaxis, :], end_idx - len(self.ref_traj), axis=0)
                p_refs = np.vstack((self.ref_traj[start_idx:], pad))
            else:
                p_refs = self.ref_traj[start_idx:end_idx]
            v_refs = np.zeros_like(p_refs)

            # Optimize control using MPC
            u_cmd, p_opt, v_opt = self.planner.optimize(p, v, p_refs, v_refs, last_u)

            # Use disturbance observer to adjust thrust command if enabled
            dist_thrust = self.observer.update(v, (u_cmd/self.planner.mpc.mass))
            thrust_command = u_cmd + dist_thrust

            # Update dynamics using the separate function
            p, v = self.simulate_dynamics(p, v, thrust_command, self.dt, self.planner.mpc.mass)

            # Log data
            history_p.append(p.copy())
            history_v.append(v.copy())
            history_ref.append(p_refs[0].copy())  # current reference point
            history_u.append(thrust_command.copy())

            # Update last command
            last_u = u_cmd

        # Convert logs to numpy arrays
        history_p = np.array(history_p)
        history_v = np.array(history_v)
        history_ref = np.array(history_ref)
        history_u = np.array(history_u)

        # Plotting results
        self.plot_results(history_p, history_ref, history_u)

    def plot_results(self, history_p, history_ref, history_u):
        time_axis = np.linspace(0, self.sim_time, len(history_p))

        # Plot position tracking
        plt.figure(figsize=(12, 6))
        for i, label in enumerate(['x', 'y', 'z']):
            plt.subplot(3, 1, i+1)
            plt.plot(time_axis, history_p[:, i], label=f'Actual {label}')
            plt.plot(time_axis, history_ref[:, i], label=f'Reference {label}', linestyle='--')
            plt.ylabel(f'Position {label} (m)')
            plt.legend()
        plt.xlabel('Time (s)')
        plt.suptitle('Drone Position Tracking')
        plt.tight_layout()
        plt.show()

        # Plot control inputs
        plt.figure(figsize=(12, 4))
        for i, label in enumerate(['fx', 'fy', 'fz']):
            plt.plot(time_axis, history_u[:, i], label=label)
        plt.ylabel('Thrust (N)')
        plt.xlabel('Time (s)')
        plt.title('Control Inputs')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    env = DroneSimEnv('mpc_test.yaml')
    env.run()
