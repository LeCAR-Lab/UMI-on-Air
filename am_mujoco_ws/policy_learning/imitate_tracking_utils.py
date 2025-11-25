import numpy as np
import os
import matplotlib.pyplot as plt
import json
from constants import DT
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d


def pad_trajectory_segment(segment: list, target_length: int, default_value: np.ndarray) -> list:
    """Pad trajectory segment to target length with last value."""
    while len(segment) < target_length:
        last = segment[-1] if segment else default_value
        segment.append(last.copy())
    return segment


def process_completed_diffusion_analysis(pending_diffusion_analysis: list, 
                                         all_actual_ee_positions: list,
                                         all_actual_ee_quaternions: list,
                                         query_frequency: int,
                                         policy,
                                         episode_tracker) -> None:
    """
    Process completed diffusion analysis periods (when full trajectory is available).
    
    Removes completed analyses from pending list.
    """
    completed_analyses = []
    for inference_timestep, start_idx in pending_diffusion_analysis:
        if len(all_actual_ee_positions) >= start_idx + query_frequency:
            actual_trajectory_segment = all_actual_ee_positions[start_idx:start_idx + query_frequency]
            actual_quat_segment = all_actual_ee_quaternions[start_idx:start_idx + query_frequency]
            save_diffusion_analysis(policy, episode_tracker, inference_timestep, 
                                   actual_trajectory_segment, actual_quat_segment)
            completed_analyses.append((inference_timestep, start_idx))
    
    for completed in completed_analyses:
        pending_diffusion_analysis.remove(completed)


def finalize_remaining_analysis(pending_diffusion_analysis: list,
                                all_actual_ee_positions: list,
                                all_actual_ee_quaternions: list,
                                query_frequency: int,
                                policy,
                                episode_tracker) -> None:
    """
    Finalize any remaining pending analysis at episode end.
    
    Pads incomplete trajectories and saves analysis for all pending inference periods.
    """
    for inference_timestep, start_idx in pending_diffusion_analysis:
        available_length = len(all_actual_ee_positions) - start_idx
        if available_length > 0:
            actual_trajectory_segment = all_actual_ee_positions[start_idx:start_idx + available_length]
            actual_quat_segment = all_actual_ee_quaternions[start_idx:start_idx + available_length]
            
            if available_length < query_frequency:
                actual_trajectory_segment = pad_trajectory_segment(
                    actual_trajectory_segment, query_frequency, np.zeros(3)
                )
                actual_quat_segment = pad_trajectory_segment(
                    actual_quat_segment, query_frequency, np.array([1,0,0,0])
                )
            
            save_diffusion_analysis(policy, episode_tracker, inference_timestep, 
                                   actual_trajectory_segment, actual_quat_segment)


class EpisodeMetricsTracker:
    """Track comprehensive metrics for each episode"""
    
    def __init__(self, episode_id, output_dir):
        self.episode_id = episode_id
        self.episode_dir = os.path.join(output_dir, f'episode_{episode_id:03d}')
        os.makedirs(self.episode_dir, exist_ok=True)
        
        # Tracking arrays
        self.timesteps = []
        self.position_mse = []
        self.orientation_distance = []
        self.mpc_costs = []
        self.inference_steps = []
        self.all_candidate_costs = []
        self.best_candidate_ids = []
        
        # QPos tracking arrays
        self.target_qpos_list = []
        self.actual_qpos_list = []
        
        # MPC prediction tracking arrays
        self.mpc_prediction_timesteps = []
        self.mpc_reference_trajectories = []
        self.mpc_predicted_trajectories = []
        
        # NEW: Trajectory comparison tracking
        self.trajectory_comparisons = []  # Per-inference trajectory error metrics
        self.reference_trajectories = []  # Store reference trajectories from diffusion
        self.actual_trajectories = []     # Store actual executed trajectories
        
        # NEW: Vanilla trajectory comparison tracking
        self.vanilla_trajectory_comparisons = []  # Per-inference vanilla trajectory error metrics
        self.vanilla_reference_trajectories = []  # Store vanilla reference trajectories from diffusion
        self.vanilla_mpc_trajectories = []        # Store vanilla MPC predicted trajectories
        
        # NEW: Main MPC tracking costs (timestep by timestep)
        self.main_mpc_tracking_costs = []  # Store main MPC tracking costs over episode
        
        # Episode summary
        self.num_timesteps = 0  # Track actual simulation timesteps instead of real time
        self.success = False
        self.episode_return = 0.0
        self.highest_reward = 0.0
        self.max_reward = 0.0
        self.crashed = False  # Track if episode crashed
        
    def add_tracking_error(self, timestep, target_pos, actual_pos, target_quat, actual_quat):
        """Add position and orientation tracking error for this timestep"""
        self.timesteps.append(timestep)
        
        # Position RMSE (Root Mean Square Error)
        pos_mse = np.mean((target_pos - actual_pos) ** 2)
        pos_rmse = np.sqrt(pos_mse)
        self.position_mse.append(pos_rmse)  # Store RMSE instead of MSE
        
        # Quaternion distance
        quat_dist = self.quaternion_distance(target_quat, actual_quat)
        self.orientation_distance.append(quat_dist)
        
    def add_mpc_costs(self, timestep, candidate_costs, best_idx, candidate_tracking_costs=None):
        """Record MPC costs for this timestep"""
        # Store in original format for compatibility
        self.inference_steps.append(timestep)
        self.all_candidate_costs.append(candidate_costs.copy())
        self.best_candidate_ids.append(best_idx)
        
        # Store in new detailed format
        self.mpc_costs.append({
            'timestep': timestep,
            'candidate_costs': candidate_costs.copy(),
            'best_idx': best_idx,
            'best_cost': candidate_costs[best_idx],
            'candidate_tracking_costs': candidate_tracking_costs.copy() if candidate_tracking_costs is not None else None,
            'best_tracking_cost': candidate_tracking_costs[best_idx] if candidate_tracking_costs is not None else None
        })
    
    def add_trajectory_comparison(self, timestep, reference_traj, actual_traj):
        """
        Add trajectory comparison data for analysis
        
        Args:
            timestep: Environment timestep when inference occurred
            reference_traj: (32, 8) final diffusion output [x,y,z,w,x,y,z,gripper] 
            actual_traj: (32, 8) actual executed trajectory
        """
        # Validate input shapes - ensure required trajectories have at least 7 columns (pos + quat + gripper)
        if reference_traj.shape[1] < 7 or actual_traj.shape[1] < 7:
            return
        
        # Interpolate reference trajectory to 32 timesteps for comparison
        if reference_traj.shape[0] == 8:
            reference_32 = self.interpolate_trajectory(reference_traj, 32)
        else:
            reference_32 = reference_traj
        
        # Ensure actual_traj has 32 timesteps
        if actual_traj.shape[0] != 32:
            if actual_traj.shape[0] < 32:
                padded_traj = np.zeros((32, actual_traj.shape[1]))
                padded_traj[:actual_traj.shape[0]] = actual_traj
                actual_traj = padded_traj
            else:
                actual_traj = actual_traj[:32]
        
        # Calculate position RMSE errors
        try:
            ref_vs_actual_pos_rmse = self.calculate_position_rmse(reference_32[:,:3], actual_traj[:,:3])
        except Exception as e:
            return
        
        # Calculate orientation distance errors with validation
        try:
            ref_vs_actual_orient_dist = self.calculate_orientation_distance(reference_32[:,3:7], actual_traj[:,3:7])
        except Exception as e:
            # Use dummy orientation distances
            ref_vs_actual_orient_dist = np.zeros(32)
        
        # Build comparison data
        comparison_data = {
            'timestep': timestep,
            'ref_vs_actual_pos_rmse': ref_vs_actual_pos_rmse.tolist(),
            'ref_vs_actual_orient_dist': ref_vs_actual_orient_dist.tolist(),
            'ref_vs_actual_pos_mean': float(np.mean(ref_vs_actual_pos_rmse)),
            'ref_vs_actual_orient_mean': float(np.mean(ref_vs_actual_orient_dist)),
        }
        
        self.trajectory_comparisons.append(comparison_data)
        
        # Store trajectories for later use
        self.reference_trajectories.append(reference_32.copy())
        self.actual_trajectories.append(actual_traj.copy())
    
    def add_vanilla_trajectory_comparison(self, timestep, vanilla_reference_traj, vanilla_mpc_traj):
        """
        Add vanilla trajectory comparison data for analysis
        
        Args:
            timestep: Environment timestep when inference occurred
            vanilla_reference_traj: (8, 8) vanilla diffusion output [x,y,z,w,x,y,z,gripper] 
            vanilla_mpc_traj: (8, 8) vanilla MPC predicted trajectory waypoints
        """
        # Validate input shapes
        if vanilla_reference_traj.shape[1] < 7 or vanilla_mpc_traj.shape[1] < 7:
            return
        
        # Interpolate trajectories to 32 timesteps for comparison
        if vanilla_reference_traj.shape[0] == 8:
            vanilla_reference_32 = self.interpolate_trajectory(vanilla_reference_traj, 32)
        else:
            vanilla_reference_32 = vanilla_reference_traj
            
        if vanilla_mpc_traj.shape[0] == 8:
            vanilla_mpc_32 = self.interpolate_trajectory(vanilla_mpc_traj, 32)
        else:
            vanilla_mpc_32 = vanilla_mpc_traj
        
        # Calculate position RMSE errors (vanilla ref vs vanilla MPC)
        try:
            vanilla_ref_vs_mpc_pos_rmse = self.calculate_position_rmse(vanilla_reference_32[:,:3], vanilla_mpc_32[:,:3])
        except Exception as e:
            return
        
        # Calculate orientation distance errors
        try:
            vanilla_ref_vs_mpc_orient_dist = self.calculate_orientation_distance(vanilla_reference_32[:,3:7], vanilla_mpc_32[:,3:7])
        except Exception as e:
            vanilla_ref_vs_mpc_orient_dist = np.zeros(32)
        
        vanilla_comparison_data = {
            'timestep': timestep,
            'vanilla_ref_vs_mpc_pos_rmse': vanilla_ref_vs_mpc_pos_rmse.tolist(),
            'vanilla_ref_vs_mpc_orient_dist': vanilla_ref_vs_mpc_orient_dist.tolist(),
            'vanilla_ref_vs_mpc_pos_mean': float(np.mean(vanilla_ref_vs_mpc_pos_rmse)),
            'vanilla_ref_vs_mpc_orient_mean': float(np.mean(vanilla_ref_vs_mpc_orient_dist))
        }
        
        self.vanilla_trajectory_comparisons.append(vanilla_comparison_data)
        
        # Store trajectories for later use
        self.vanilla_reference_trajectories.append(vanilla_reference_32.copy())
        self.vanilla_mpc_trajectories.append(vanilla_mpc_32.copy())
        
    def add_main_mpc_tracking_cost(self, timestep, tracking_cost):
        """Add main MPC tracking cost for this timestep"""
        self.main_mpc_tracking_costs.append({
            'timestep': timestep,
            'tracking_cost': tracking_cost
        })
        
    def calculate_position_rmse(self, traj1_pos, traj2_pos):
        """Calculate position RMSE between two position trajectories"""
        if traj1_pos.shape != traj2_pos.shape:
            raise ValueError(f"Trajectory shapes don't match: {traj1_pos.shape} vs {traj2_pos.shape}")
        
        # Calculate RMSE per timestep
        squared_errors = (traj1_pos - traj2_pos) ** 2
        mse_per_timestep = np.mean(squared_errors, axis=1)  # Mean across x,y,z for each timestep
        rmse_per_timestep = np.sqrt(mse_per_timestep)
        return rmse_per_timestep
        
    def calculate_orientation_distance(self, traj1_quat, traj2_quat):
        """Calculate quaternion distance between two quaternion trajectories"""
        if traj1_quat.shape != traj2_quat.shape:
            raise ValueError(f"Quaternion trajectory shapes don't match: {traj1_quat.shape} vs {traj2_quat.shape}")
            
        distances = []
        for i in range(len(traj1_quat)):
            dist = self.quaternion_distance(traj1_quat[i], traj2_quat[i])
            distances.append(dist)
        return np.array(distances)
        
    def interpolate_trajectory(self, waypoints, target_horizon=32):
        """
        Interpolate a sequence of 8-D waypoints to the desired horizon length.
        
        Args:
            waypoints: np.ndarray with shape (N, 8) where N ≥ 2.
            target_horizon: Desired number of steps in the returned trajectory.
            
        Returns:
            np.ndarray with shape (target_horizon, 8)
        """
        from scipy.spatial.transform import Rotation as R, Slerp
        
        # Early exit if the trajectory is already at the desired resolution
        if waypoints.shape[0] == target_horizon:
            return waypoints.copy()

        interpolated = []
        n_segments = waypoints.shape[0] - 1

        for i in range(target_horizon):
            # Normalised parameter along entire trajectory [0, 1]
            t_global = i / (target_horizon - 1) if target_horizon > 1 else 0.0
            segment_length = 1.0 / n_segments
            segment_idx = min(int(t_global / segment_length), n_segments - 1)
            # Local parameter within current segment [0, 1]
            t_local = (t_global - segment_idx * segment_length) / segment_length
            t_local = np.clip(t_local, 0.0, 1.0)

            start_action = waypoints[segment_idx]
            end_action = waypoints[segment_idx + 1]

            # Linear interpolation for position
            pos = start_action[:3] + t_local * (end_action[:3] - start_action[:3])

            # Linear interpolation for gripper state
            gripper = start_action[7] + t_local * (end_action[7] - start_action[7])

            # SLERP for orientation (w, x, y, z) → (x, y, z, w) for SciPy
            start_qwxyz = start_action[3:7]
            end_qwxyz = end_action[3:7]
            start_qxyzw = np.array([start_qwxyz[1], start_qwxyz[2], start_qwxyz[3], start_qwxyz[0]])
            end_qxyzw = np.array([end_qwxyz[1], end_qwxyz[2], end_qwxyz[3], end_qwxyz[0]])

            start_rot = R.from_quat(start_qxyzw)
            end_rot = R.from_quat(end_qxyzw)
            slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
            interp_rot = slerp(t_local)
            interp_qxyzw = interp_rot.as_quat()
            interp_qwxyz = np.array([interp_qxyzw[3], interp_qxyzw[0], interp_qxyzw[1], interp_qxyzw[2]])

            interpolated.append(np.concatenate([pos, interp_qwxyz, [gripper]], axis=-1))

        return np.asarray(interpolated, dtype=waypoints.dtype)
        
    def add_qpos_data(self, target_qpos, actual_qpos):
        """Add target and actual QPos data for this timestep"""
        self.target_qpos_list.append(target_qpos.copy())
        self.actual_qpos_list.append(actual_qpos.copy())
        
    def add_mpc_prediction(self, timestep, reference_trajectory, predicted_trajectory):
        """Add MPC prediction data for later plotting when actual trajectory becomes available"""
        # Store the data for later plotting when actual trajectory is available
        self.mpc_prediction_timesteps.append(timestep)
        self.mpc_reference_trajectories.append(reference_trajectory.copy())
        self.mpc_predicted_trajectories.append(predicted_trajectory.copy())
        
        # Note: Actual plotting will be done via plot_mpc_prediction_with_actual() 
        # when the actual trajectory becomes available
        
    def plot_mpc_prediction_with_actual(self, timestep, reference_traj, predicted_traj, actual_pos_traj, actual_quat_traj=None):
        """Generate MPC prediction plots with actual trajectory data"""
        if reference_traj is None or predicted_traj is None:
            return
            
        # Create inference directory structure matching diffusion analysis
        inference_dir = os.path.join(self.episode_dir, 'inference', f'step_{timestep:04d}')
        os.makedirs(inference_dir, exist_ok=True)
        
        # Extract positions from trajectories
        ref_positions = reference_traj[:, :3]  # Reference: (32, 3)
        pred_positions = predicted_traj[:, :3]  # MPC predicted: (N+1, 3)
        
        # Create 3D trajectory plot
        fig = plt.figure(figsize=(15, 12))
        
        # 3D plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], 
                'b-', linewidth=2, label='Reference Trajectory', alpha=0.8)
        ax1.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 
                'r--', linewidth=2, label='MPC Predicted', alpha=0.8)
        
        # Plot actual trajectory if available
        if actual_pos_traj is not None and len(actual_pos_traj) == 32:
            actual_positions = np.array([pos[:3] for pos in actual_pos_traj])
            ax1.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 
                    'g-', linewidth=3, label='Actual Executed', alpha=0.9)
        
        # Mark start and end points
        ax1.scatter(*ref_positions[0], color='green', s=100, label='Start', marker='o')
        ax1.scatter(*ref_positions[-1], color='orange', s=100, label='End', marker='s')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D Trajectory Comparison (t={timestep})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Individual axis plots
        axes = ['X', 'Y', 'Z']
        for j, axis in enumerate(axes):
            ax = fig.add_subplot(2, 3, j + 2)
            
            # Plot reference and predicted trajectories
            ax.plot(range(len(ref_positions)), ref_positions[:, j], 
                   'b-', linewidth=2, label='Reference', alpha=0.8)
            ax.plot(range(len(pred_positions)), pred_positions[:, j], 
                   'r--', linewidth=2, label='MPC Predicted', alpha=0.8)
            
            # Plot actual trajectory if available
            if actual_pos_traj is not None and len(actual_pos_traj) == 32:
                actual_positions = np.array([pos[:3] for pos in actual_pos_traj])
                ax.plot(range(32), actual_positions[:, j], 
                       'g-', linewidth=3, label='Actual Executed', alpha=0.9)
            
            ax.set_xlabel('Waypoint Index')
            ax.set_ylabel(f'{axis} Position (m)')
            ax.set_title(f'{axis}-axis Trajectory (t={timestep})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Position error plot
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Calculate position errors (interpolate if different lengths)
        min_len = min(len(ref_positions), len(pred_positions))
        if len(ref_positions) != len(pred_positions):
            # Interpolate to match lengths
            if len(ref_positions) > len(pred_positions):
                # Interpolate predicted to match reference
                pred_interp = interp1d(np.linspace(0, 1, len(pred_positions)), 
                                     pred_positions, axis=0, kind='linear')
                pred_positions_matched = pred_interp(np.linspace(0, 1, len(ref_positions)))
                ref_positions_matched = ref_positions
            else:
                # Interpolate reference to match predicted
                ref_interp = interp1d(np.linspace(0, 1, len(ref_positions)), 
                                    ref_positions, axis=0, kind='linear')
                ref_positions_matched = ref_interp(np.linspace(0, 1, len(pred_positions)))
                pred_positions_matched = pred_positions
        else:
            ref_positions_matched = ref_positions
            pred_positions_matched = pred_positions
        
        # Calculate position errors
        position_errors = np.linalg.norm(pred_positions_matched - ref_positions_matched, axis=1)
        
        ax5.plot(range(len(position_errors)), position_errors, 
                'g-', linewidth=2, alpha=0.8, label='MPC vs Reference')
        
        # Calculate actual vs reference errors if actual trajectory is available
        if actual_pos_traj is not None and len(actual_pos_traj) == 32:
            actual_positions = np.array([pos[:3] for pos in actual_pos_traj])
            actual_ref_errors = np.linalg.norm(actual_positions - ref_positions, axis=1)
            ax5.plot(range(len(actual_ref_errors)), actual_ref_errors, 
                    'b-', linewidth=2, alpha=0.8, label='Actual vs Reference')
            
            # Calculate actual vs predicted errors
            if len(actual_positions) == len(pred_positions_matched):
                actual_pred_errors = np.linalg.norm(actual_positions - pred_positions_matched, axis=1)
                ax5.plot(range(len(actual_pred_errors)), actual_pred_errors, 
                        'r-', linewidth=2, alpha=0.8, label='Actual vs MPC')
        
        ax5.set_xlabel('Waypoint Index')
        ax5.set_ylabel('Position Error (m)')
        ax5.set_title(f'Position Error Analysis (t={timestep})')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add error statistics
        mean_error = np.mean(position_errors)
        max_error = np.max(position_errors)
        ax5.axhline(mean_error, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean MPC Error: {mean_error:.4f}m')
        
        # Summary statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f'MPC Prediction Statistics (t={timestep})\n\n'
        stats_text += f'Reference points: {len(ref_positions)}\n'
        stats_text += f'Predicted points: {len(pred_positions)}\n'
        stats_text += f'Mean MPC error: {mean_error:.4f} m\n'
        stats_text += f'Max MPC error: {max_error:.4f} m\n'
        stats_text += f'RMS MPC error: {np.sqrt(np.mean(position_errors**2)):.4f} m\n'
        
        if actual_pos_traj is not None and len(actual_pos_traj) == 32:
            actual_positions = np.array([pos[:3] for pos in actual_pos_traj])
            actual_ref_errors = np.linalg.norm(actual_positions - ref_positions, axis=1)
            stats_text += f'\nActual vs Reference:\n'
            stats_text += f'Mean error: {np.mean(actual_ref_errors):.4f} m\n'
            stats_text += f'Max error: {np.max(actual_ref_errors):.4f} m\n'
            stats_text += f'RMS error: {np.sqrt(np.mean(actual_ref_errors**2)):.4f} m\n'
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Episode {self.episode_id} - MPC Prediction Analysis (Timestep {timestep})', 
                    fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(inference_dir, 'mpc_prediction_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved MPC prediction plot to {plot_path}')
        
        # ------------------------------------------------------------
        # Quaternion Component Comparison (w, x, y, z)
        # ------------------------------------------------------------
        ref_quats = reference_traj[:, 3:7]  # (32, 4)
        pred_quats = predicted_traj[:, 3:7]  # (N+1, 4)

        # Interpolate predicted quats to match reference length if necessary
        if len(pred_quats) != len(ref_quats):
            pred_quat_interp = interp1d(np.linspace(0, 1, len(pred_quats)), pred_quats, axis=0, kind='linear')
            pred_quats_matched = pred_quat_interp(np.linspace(0, 1, len(ref_quats)))
        else:
            pred_quats_matched = pred_quats

        comp_names = ['w', 'x', 'y', 'z']
        fig_q = plt.figure(figsize=(12, 8))
        for idx, comp in enumerate(comp_names):
            ax_q = fig_q.add_subplot(2, 2, idx + 1)

            ax_q.plot(range(len(ref_quats)), ref_quats[:, idx], 'b-', linewidth=2, label='Reference', alpha=0.8)
            ax_q.plot(range(len(pred_quats_matched)), pred_quats_matched[:, idx], 'r--', linewidth=2, label='MPC Predicted', alpha=0.8)

            # Plot actual quaternion trajectory if available
            if actual_quat_traj is not None and len(actual_quat_traj) == 32:
                actual_quats = np.array(actual_quat_traj)
                ax_q.plot(range(32), actual_quats[:, idx], 'g-', linewidth=2, label='Actual Executed', alpha=0.9)

            ax_q.set_xlabel('Waypoint Index')
            ax_q.set_ylabel(f'{comp}')
            ax_q.set_title(f'Quaternion {comp}-component (t={timestep})')
            ax_q.grid(True, alpha=0.3)
            if idx == 0:
                ax_q.legend()

        plt.suptitle(f'Episode {self.episode_id} - Quaternion Comparison (Timestep {timestep})', fontsize=16)
        plt.tight_layout()
        quat_plot_path = os.path.join(inference_dir, 'mpc_quaternion_comparison.png')
        plt.savefig(quat_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved MPC quaternion plot to {quat_plot_path}')
        
    def _plot_mpc_prediction_realtime(self, timestep, reference_traj, predicted_traj):
        """Generate MPC prediction plots in real-time using the same directory structure as diffusion analysis"""
        # This method is now deprecated - use plot_mpc_prediction_with_actual instead
        pass
        
    def finalize_episode(self, max_reward, crashed=False, failure_reasons=None, 
                         success=False, episode_return=0.0, highest_reward=0.0):
        """Calculate final episode metrics
        
        Args:
            max_reward: Maximum possible reward for the task
            crashed: Whether the episode crashed due to simulation error
            failure_reasons: List of failure reasons (e.g., can dropped)
            success: Whether the episode was successful (computed in main loop for consistency)
            episode_return: Sum of all rewards (computed in main loop)
            highest_reward: Maximum reward achieved (computed in main loop)
        """
        # Calculate simulation time based on timesteps executed
        self.episode_duration = self.num_timesteps * DT  # Simulation time, not real time
        self.max_reward = max_reward
        self.crashed = crashed  # Store crash status
        self.failure_reasons = failure_reasons or []  # Store failure reasons
        
        # Debug print to verify calculation
        print(f"Episode duration calculation: {self.num_timesteps} timesteps × {DT} s/timestep = {self.episode_duration:.2f} s")
        
        # Use values from main loop (single source of truth for reward-based metrics)
        self.episode_return = episode_return
        self.highest_reward = highest_reward
        self.success = success
        
        # Calculate average tracking errors
        self.avg_position_rmse = np.mean(self.position_mse) if self.position_mse else 0.0  # Now RMSE
        self.avg_orientation_distance = np.mean(self.orientation_distance) if self.orientation_distance else 0.0
        
        # Calculate trajectory comparison averages
        if self.trajectory_comparisons:
            self.avg_ref_vs_actual_pos_rmse = np.mean([comp['ref_vs_actual_pos_mean'] for comp in self.trajectory_comparisons])
            self.avg_ref_vs_mpc_pos_rmse = np.mean([comp['ref_vs_mpc_pos_mean'] for comp in self.trajectory_comparisons])
            self.avg_mpc_vs_actual_pos_rmse = np.mean([comp['mpc_vs_actual_pos_mean'] for comp in self.trajectory_comparisons])
            self.avg_ref_vs_actual_orient_dist = np.mean([comp['ref_vs_actual_orient_mean'] for comp in self.trajectory_comparisons])
            self.avg_ref_vs_mpc_orient_dist = np.mean([comp['ref_vs_mpc_orient_mean'] for comp in self.trajectory_comparisons])
            self.avg_mpc_vs_actual_orient_dist = np.mean([comp['mpc_vs_actual_orient_mean'] for comp in self.trajectory_comparisons])
        else:
            self.avg_ref_vs_actual_pos_rmse = 0.0
            self.avg_ref_vs_mpc_pos_rmse = 0.0
            self.avg_mpc_vs_actual_pos_rmse = 0.0
            self.avg_ref_vs_actual_orient_dist = 0.0
            self.avg_ref_vs_mpc_orient_dist = 0.0
            self.avg_mpc_vs_actual_orient_dist = 0.0
            
        # Calculate vanilla trajectory comparison averages
        if self.vanilla_trajectory_comparisons:
            self.avg_vanilla_ref_vs_mpc_pos_rmse = np.mean([comp['vanilla_ref_vs_mpc_pos_mean'] for comp in self.vanilla_trajectory_comparisons])
            self.avg_vanilla_ref_vs_mpc_orient_dist = np.mean([comp['vanilla_ref_vs_mpc_orient_mean'] for comp in self.vanilla_trajectory_comparisons])
        else:
            self.avg_vanilla_ref_vs_mpc_pos_rmse = 0.0
            self.avg_vanilla_ref_vs_mpc_orient_dist = 0.0
            
        # Calculate main MPC tracking cost average
        if self.main_mpc_tracking_costs:
            self.avg_main_mpc_tracking_cost = np.mean([cost['tracking_cost'] for cost in self.main_mpc_tracking_costs])
        else:
            self.avg_main_mpc_tracking_cost = 0.0
        
    def quaternion_distance(self, q1, q2):
        """Calculate quaternion distance between two quaternions"""
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1) if np.linalg.norm(q1) > 1e-6 else np.array([1, 0, 0, 0])
        q2 = q2 / np.linalg.norm(q2) if np.linalg.norm(q2) > 1e-6 else np.array([1, 0, 0, 0])
        
        # Calculate dot product (cosine of half angle)
        dot_product = np.abs(np.dot(q1, q2))
        dot_product = np.clip(dot_product, 0.0, 1.0)  # Numerical stability
        
        # Convert to angular distance in radians
        angular_distance = 2 * np.arccos(dot_product)
        return angular_distance
        
    def save_metrics(self):
        """Save all metrics and create plots for this episode"""
        # Save metrics to JSON
        metrics_data = {
            'episode_id': self.episode_id,
            'success': bool(self.success),
            'crashed': bool(self.crashed),  # Add crash status
            'failure_reasons': self.failure_reasons,  # Add failure reasons
            'episode_return': float(self.episode_return),
            'highest_reward': float(self.highest_reward),
            'max_reward': float(self.max_reward),
            'episode_duration': float(self.episode_duration),
            'avg_position_rmse': float(self.avg_position_rmse),  # Changed from MSE to RMSE
            'avg_orientation_distance': float(self.avg_orientation_distance),
            'avg_ref_vs_actual_pos_rmse': float(self.avg_ref_vs_actual_pos_rmse),
            'avg_ref_vs_mpc_pos_rmse': float(self.avg_ref_vs_mpc_pos_rmse),
            'avg_mpc_vs_actual_pos_rmse': float(self.avg_mpc_vs_actual_pos_rmse),
            'avg_ref_vs_actual_orient_dist': float(self.avg_ref_vs_actual_orient_dist),
            'avg_ref_vs_mpc_orient_dist': float(self.avg_ref_vs_mpc_orient_dist),
            'avg_mpc_vs_actual_orient_dist': float(self.avg_mpc_vs_actual_orient_dist),
            'avg_vanilla_ref_vs_mpc_pos_rmse': float(self.avg_vanilla_ref_vs_mpc_pos_rmse),
            'avg_vanilla_ref_vs_mpc_orient_dist': float(self.avg_vanilla_ref_vs_mpc_orient_dist),
            'avg_main_mpc_tracking_cost': float(self.avg_main_mpc_tracking_cost),
            'timestep_data': {
                'timesteps': self.timesteps,
                'position_rmse': [float(x) for x in self.position_mse],  # Changed from MSE to RMSE
                'orientation_distance': [float(x) for x in self.orientation_distance]
            },
            'mpc_data': {
                'inference_steps': self.inference_steps,
                'all_candidate_costs': [[float(c) for c in costs] for costs in self.all_candidate_costs],
                'best_candidate_ids': self.best_candidate_ids
            },
            'qpos_data': {
                'target_qpos': [qpos.tolist() if hasattr(qpos, 'tolist') else list(qpos) for qpos in self.target_qpos_list],
                'actual_qpos': [qpos.tolist() if hasattr(qpos, 'tolist') else list(qpos) for qpos in self.actual_qpos_list]
            },
            'trajectory_comparison_data': {
                'trajectory_comparisons': self.trajectory_comparisons,
                'reference_trajectories': [traj.tolist() for traj in self.reference_trajectories],
                'actual_trajectories': [traj.tolist() for traj in self.actual_trajectories]
            },
            'vanilla_trajectory_comparison_data': {
                'vanilla_trajectory_comparisons': self.vanilla_trajectory_comparisons,
                'vanilla_reference_trajectories': [traj.tolist() for traj in self.vanilla_reference_trajectories],
                'vanilla_mpc_trajectories': [traj.tolist() for traj in self.vanilla_mpc_trajectories]
            },
            'main_mpc_tracking_costs': self.main_mpc_tracking_costs
        }
        
        metrics_path = os.path.join(self.episode_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Create tracking error plot
        self.plot_tracking_errors()
        
        # Create MPC costs plot if available
        if self.inference_steps:
            self.plot_mpc_costs()
            
            # Create tracking vs total cost comparison if meaningful tracking data is available
            if any(cost_data.get('candidate_tracking_costs') is not None and 
                   any(tc > 0 for tc in cost_data['candidate_tracking_costs']) 
                   for cost_data in self.mpc_costs):
                self.plot_tracking_vs_total_cost()
                
        # Create QPos comparison plots
        self.plot_qpos_comparison()
        
        # NEW: Create trajectory comparison episode summary
        if self.trajectory_comparisons:
            self.plot_episode_trajectory_summary()
            
        # NEW: Create main MPC tracking cost plot
        if self.main_mpc_tracking_costs:
            self.plot_main_mpc_tracking_costs()
        
        # MPC prediction plots are now generated in real-time, no need for batch plotting
            
        return metrics_data
        
    def plot_tracking_errors(self):
        """Create tracking error plots for position and orientation"""
        if not self.timesteps:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Position RMSE plot
        ax1.plot(self.timesteps, self.position_mse, 'b-', linewidth=2, label='Position RMSE')
        ax1.set_ylabel('Position RMSE (m)', fontsize=12)
        ax1.set_title(f'Episode {self.episode_id} - Tracking Errors\nAvg Position RMSE: {self.avg_position_rmse:.6f} m', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Orientation distance plot
        ax2.plot(self.timesteps, self.orientation_distance, 'r-', linewidth=2, label='Orientation Distance')
        ax2.set_xlabel('Timestep', fontsize=12)
        ax2.set_ylabel('Orientation Distance (rad)', fontsize=12)
        ax2.set_title(f'Avg Orientation Distance: {self.avg_orientation_distance:.6f} rad', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.episode_dir, 'tracking_errors.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_mpc_costs(self):
        """Create MPC costs plot"""
        if not self.inference_steps:
            return
            
        plt.figure(figsize=(12, 6))
        
        for idx, (step_idx, costs, best_i) in enumerate(zip(self.inference_steps, self.all_candidate_costs, self.best_candidate_ids)):
            costs_np = np.array(costs)
            finite_mask = np.isfinite(costs_np)
            range_mask = (costs_np >= 0.0) & (costs_np <= 1.0)
            plot_mask = finite_mask & range_mask
            
            # Scatter all costs in valid range
            if np.any(plot_mask):
                plt.scatter(np.full(np.sum(plot_mask), step_idx), costs_np[plot_mask], 
                           color='blue', s=20, alpha=0.6)
            
            # Highlight chosen trajectory
            chosen_cost = costs_np[best_i]
            if np.isfinite(chosen_cost) and (0.0 <= chosen_cost <= 1.0):
                plt.scatter(step_idx, chosen_cost, color='red', marker='*', s=120, 
                           label='Selected' if idx == 0 else "")
            
            # Vertical guide line
            plt.axvline(step_idx, color='grey', alpha=0.2, linewidth=0.5)
        
        plt.xlabel('Environment Step', fontsize=12)
        plt.ylabel('MPC Cost', fontsize=12)
        plt.title(f'Episode {self.episode_id} - MPC Trajectory Costs', fontsize=14)
        plt.grid(True, alpha=0.3)
        if self.inference_steps:
            plt.legend()
        
        # Set reasonable y-axis limits
        all_costs_flat = np.concatenate([
            np.array(costs)[np.isfinite(costs) & (np.array(costs) >= 0.0)]
            for costs in self.all_candidate_costs
        ])
        if all_costs_flat.size > 0:
            lower_bound = np.min(all_costs_flat)
            upper_bound = np.percentile(all_costs_flat, 95)
            margin = (upper_bound - lower_bound) * 0.05
            plt.ylim(max(0.0, lower_bound - margin), upper_bound + margin)
        
        plt.tight_layout()
        plot_path = os.path.join(self.episode_dir, 'mpc_costs.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_tracking_vs_total_cost(self):
        """Create tracking vs total cost comparison plot"""
        if not self.mpc_costs:
            return
            
        # Extract tracking and total costs for selected trajectories
        tracking_costs = []
        total_costs = []
        timesteps = []
        
        for cost_data in self.mpc_costs:
            if (cost_data.get('best_tracking_cost') is not None and 
                cost_data.get('best_cost') is not None):
                tracking_costs.append(cost_data['best_tracking_cost'])
                total_costs.append(cost_data['best_cost'])
                timesteps.append(cost_data['timestep'])
        
        if not tracking_costs:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Time series comparison
        ax1.plot(timesteps, tracking_costs, 'b-', linewidth=2, label='Tracking Cost', alpha=0.8)
        ax1.plot(timesteps, total_costs, 'r-', linewidth=2, label='Total Cost', alpha=0.8)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('MPC Cost')
        ax1.set_title('MPC Cost Comparison Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot: tracking vs total
        ax2.scatter(tracking_costs, total_costs, alpha=0.6, s=30)
        ax2.set_xlabel('Tracking Cost')
        ax2.set_ylabel('Total Cost')
        ax2.set_title('Tracking vs Total Cost Correlation')
        ax2.grid(True, alpha=0.3)
        
        # Add regression/trend line
        if len(tracking_costs) > 1:
            # Force regression through origin (y = mx, no intercept)
            X = np.array(tracking_costs).reshape(-1, 1)
            y = np.array(total_costs)
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            x_trend = np.linspace(min(tracking_costs), max(tracking_costs), 100)
            y_trend = reg.predict(x_trend.reshape(-1, 1))
            slope = reg.coef_[0]
            ax2.plot(x_trend, y_trend, '--', color='grey', alpha=0.8, linewidth=2, 
                    label=f'Trend y={slope:.2f}x (R²={reg.score(X, y):.3f})')
            ax2.legend()
        
        # 3. Cost difference over time
        cost_diff = np.array(total_costs) - np.array(tracking_costs)
        ax3.plot(timesteps, cost_diff, 'g-', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Total Cost - Tracking Cost')
        ax3.set_title('Non-Tracking Cost Components Over Time\n(Control Effort + Stability + Regularization)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # 4. Distribution comparison
        ax4.hist(tracking_costs, bins=20, alpha=0.6, label='Tracking Cost', color='blue', density=True)
        ax4.hist(total_costs, bins=20, alpha=0.6, label='Total Cost', color='red', density=True)
        ax4.set_xlabel('Cost Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Cost Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        avg_tracking = np.mean(tracking_costs)
        avg_total = np.mean(total_costs)
        avg_diff = np.mean(cost_diff)
        correlation = np.corrcoef(tracking_costs, total_costs)[0, 1]
        
        stats_text = f'Statistics:\n'
        stats_text += f'Avg Tracking: {avg_tracking:.4f}\n'
        stats_text += f'Avg Total: {avg_total:.4f}\n'
        stats_text += f'Avg Difference: {avg_diff:.4f}\n'
        stats_text += f'Correlation: {correlation:.3f}'
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Episode {self.episode_id} - MPC Cost Analysis', fontsize=16)
        plt.tight_layout()
        plot_path = os.path.join(self.episode_dir, 'tracking_vs_total_cost.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_qpos_comparison(self):
        """Create QPos comparison plots for target vs actual positions"""
        if not self.target_qpos_list or not self.actual_qpos_list:
            return
            
        target_qpos_array = np.array(self.target_qpos_list)
        actual_qpos_array = np.array(self.actual_qpos_list)
        
        print(f"Target QPos shape: {target_qpos_array.shape}")
        print(f"Actual QPos shape: {actual_qpos_array.shape}")
        
        # Plot 1: Target QPos dimensions (first 3 dimensions) - exactly like original
        print("shape target_qpos_list: ", target_qpos_array.shape)
        plt.figure(figsize=(10, 6))
        for i in range(min(3, target_qpos_array.shape[1])):  # Plot the first three dimensions
            plt.plot(target_qpos_array[:, i], label=f'Dimension {i+1}')
        plt.xlabel('Timestep')
        plt.ylabel('Target QPos')
        plt.title(f'Target QPos Dimensions for Episode {self.episode_id}')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.episode_dir, 'target_qpos.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Saved target_qpos_list plot to {plot_path}')
        
        # Plot 2: QPos comparison (actual vs target) for first 3 dimensions - exactly like original
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(min(3, target_qpos_array.shape[1])):
            ax[i].plot(actual_qpos_array[:, i])  # actual qpos (no label in original)
            ax[i].plot(target_qpos_array[:, i])  # target qpos (no label in original)
            ax[i].set_title(f'QPos Dimension {i}')

        plt.xlabel('Timestep')
        plt.ylabel('QPos')
        plt.title(f'QPos Dimensions for Episode {self.episode_id}')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.episode_dir, 'qpos_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        print(f'Saved qpos_list plot to {plot_path}')

    def plot_trajectory_comparison_analysis(self, timestep, reference_traj, actual_traj, inference_dir, vanilla_reference_traj=None):
        """
        Create focused trajectory comparison plot showing:
        - Reference (guided diffusion) vs Actual trajectories
        - Vanilla diffusion trajectory (if available)
        - Position and orientation error analysis
        
        Args:
            timestep: Environment timestep
            reference_traj: (32, 8) guided reference trajectory from diffusion
            actual_traj: (32, 8) actual executed trajectory
            inference_dir: Directory to save the plot
            vanilla_reference_traj: (8, 8) vanilla reference trajectory from diffusion (optional)
        """
        # Interpolate reference trajectory to 32 timesteps for comparison
        if reference_traj.shape[0] == 8:
            reference_32 = self.interpolate_trajectory(reference_traj, 32)
        else:
            reference_32 = reference_traj
            
        # Interpolate vanilla trajectory to 32 timesteps if available
        vanilla_reference_32 = None
        if vanilla_reference_traj is not None:
            if vanilla_reference_traj.shape[0] == 8:
                vanilla_reference_32 = self.interpolate_trajectory(vanilla_reference_traj, 32)
            else:
                vanilla_reference_32 = vanilla_reference_traj
        
        # Calculate trajectory errors  
        ref_vs_actual_pos_rmse = self.calculate_position_rmse(reference_32[:,:3], actual_traj[:,:3])
        ref_vs_actual_orient_dist = self.calculate_orientation_distance(reference_32[:,3:7], actual_traj[:,3:7])
        
        # Create 2x3 plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: X, Y, Z trajectory comparisons
        coords = ['X', 'Y', 'Z']
        for i, coord in enumerate(coords):
            ax = axes[0, i]
            
            # Plot trajectories
            ax.plot(range(32), reference_32[:, i], 'b-', linewidth=3, alpha=0.8, label='Reference')
            ax.plot(range(32), actual_traj[:, i], 'g-', linewidth=2, alpha=0.8, label='Actual Executed')
            
            # Plot vanilla trajectory if available
            if vanilla_reference_32 is not None:
                ax.plot(range(32), vanilla_reference_32[:, i], 'c-', linewidth=2, alpha=0.7, label='Vanilla Reference')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel(f'{coord} Position (m)')
            ax.set_title(f'{coord}-axis Trajectory Comparison')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=9)
        
        # Bottom left: Position errors over time
        ax = axes[1, 0]
        ax.plot(range(32), ref_vs_actual_pos_rmse, 'b-', linewidth=2, label='Reference vs Actual', alpha=0.8)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position RMSE (m)')
        ax.set_title('Position Error Over Time')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Bottom middle: Orientation errors over time
        ax = axes[1, 1]
        ax.plot(range(32), ref_vs_actual_orient_dist, 'b-', linewidth=2, label='Reference vs Actual', alpha=0.8)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Orientation Distance (rad)')
        ax.set_title('Orientation Error Over Time')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Bottom right: Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate summary statistics
        stats_text = f'Trajectory Analysis Summary (t={timestep})\n\n'
        
        # Trajectory statistics
        stats_text += 'TRAJECTORY ERRORS:\n'
        stats_text += 'Position RMSE (m):\n'
        stats_text += f'  Reference vs Actual: {np.mean(ref_vs_actual_pos_rmse):.4f} ± {np.std(ref_vs_actual_pos_rmse):.4f}\n'
        stats_text += '\n'
        
        stats_text += 'Orientation Distance (rad):\n'
        stats_text += f'  Reference vs Actual: {np.mean(ref_vs_actual_orient_dist):.4f} ± {np.std(ref_vs_actual_orient_dist):.4f}\n'
        stats_text += '\n'
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'Episode {self.episode_id} - 5-Trajectory Comparison Analysis (Timestep {timestep})', 
                    fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(inference_dir, 'trajectory_comparison_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved trajectory comparison analysis to {plot_path}')
        
    def plot_episode_trajectory_summary(self):
        """Create episode-level trajectory error analysis plots"""
        if not self.trajectory_comparisons:
            return
            
        # Extract data for plotting
        timesteps = [comp['timestep'] for comp in self.trajectory_comparisons]
        ref_vs_actual_pos = [comp['ref_vs_actual_pos_mean'] for comp in self.trajectory_comparisons]
        ref_vs_mpc_pos = [comp['ref_vs_mpc_pos_mean'] for comp in self.trajectory_comparisons]
        mpc_vs_actual_pos = [comp['mpc_vs_actual_pos_mean'] for comp in self.trajectory_comparisons]
        
        ref_vs_actual_orient = [comp['ref_vs_actual_orient_mean'] for comp in self.trajectory_comparisons]
        ref_vs_mpc_orient = [comp['ref_vs_mpc_orient_mean'] for comp in self.trajectory_comparisons]
        mpc_vs_actual_orient = [comp['mpc_vs_actual_orient_mean'] for comp in self.trajectory_comparisons]
        
        # Create 2x2 plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Position errors over episode
        ax1.plot(timesteps, ref_vs_actual_pos, 'b-', linewidth=2, marker='o', markersize=4, 
                label=f'Ref vs Actual (avg: {np.mean(ref_vs_actual_pos):.4f})', alpha=0.8)
        ax1.plot(timesteps, ref_vs_mpc_pos, 'r-', linewidth=2, marker='s', markersize=4,
                label=f'Ref vs MPC (avg: {np.mean(ref_vs_mpc_pos):.4f})', alpha=0.8)
        ax1.plot(timesteps, mpc_vs_actual_pos, 'g-', linewidth=2, marker='^', markersize=4,
                label=f'MPC vs Actual (avg: {np.mean(mpc_vs_actual_pos):.4f})', alpha=0.8)
        ax1.set_xlabel('Episode Timestep')
        ax1.set_ylabel('Position RMSE (m)')
        ax1.set_title('Position Errors Throughout Episode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Orientation errors over episode
        ax2.plot(timesteps, ref_vs_actual_orient, 'b-', linewidth=2, marker='o', markersize=4,
                label=f'Ref vs Actual (avg: {np.mean(ref_vs_actual_orient):.4f})', alpha=0.8)
        ax2.plot(timesteps, ref_vs_mpc_orient, 'r-', linewidth=2, marker='s', markersize=4,
                label=f'Ref vs MPC (avg: {np.mean(ref_vs_mpc_orient):.4f})', alpha=0.8)
        ax2.plot(timesteps, mpc_vs_actual_orient, 'g-', linewidth=2, marker='^', markersize=4,
                label=f'MPC vs Actual (avg: {np.mean(mpc_vs_actual_orient):.4f})', alpha=0.8)
        ax2.set_xlabel('Episode Timestep')
        ax2.set_ylabel('Orientation Distance (rad)')
        ax2.set_title('Orientation Errors Throughout Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Position error distributions
        ax3.hist([ref_vs_actual_pos, ref_vs_mpc_pos, mpc_vs_actual_pos], 
                bins=min(10, len(timesteps)), alpha=0.7, 
                label=['Ref vs Actual', 'Ref vs MPC', 'MPC vs Actual'],
                color=['blue', 'red', 'green'])
        ax3.set_xlabel('Position RMSE (m)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Position Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Orientation error distributions
        ax4.hist([ref_vs_actual_orient, ref_vs_mpc_orient, mpc_vs_actual_orient], 
                bins=min(10, len(timesteps)), alpha=0.7,
                label=['Ref vs Actual', 'Ref vs MPC', 'MPC vs Actual'],
                color=['blue', 'red', 'green'])
        ax4.set_xlabel('Orientation Distance (rad)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Orientation Error Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Episode {self.episode_id} - Trajectory Error Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.episode_dir, 'episode_trajectory_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved episode trajectory summary to {plot_path}')
        
    def plot_main_mpc_tracking_costs(self):
        """Create main MPC tracking cost analysis plot"""
        if not self.main_mpc_tracking_costs:
            return
            
        timesteps = [cost['timestep'] for cost in self.main_mpc_tracking_costs]
        tracking_costs = [cost['tracking_cost'] for cost in self.main_mpc_tracking_costs]
        
        # Create 2x2 plot for comprehensive analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Main tracking cost over time
        ax1.plot(timesteps, tracking_costs, 'purple', linewidth=2, marker='o', markersize=3, alpha=0.8)
        ax1.set_xlabel('Episode Timestep')
        ax1.set_ylabel('Main MPC Tracking Cost')
        ax1.set_title(f'Main MPC Tracking Cost Over Episode\nAvg: {np.mean(tracking_costs):.6f}')
        ax1.grid(True, alpha=0.3)
        
        # Tracking cost distribution
        ax2.hist(tracking_costs, bins=min(20, len(tracking_costs)), alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(np.mean(tracking_costs), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(tracking_costs):.6f}')
        ax2.set_xlabel('Main MPC Tracking Cost')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Tracking Cost Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling average (if enough data points)
        if len(tracking_costs) > 5:
            window_size = min(10, len(tracking_costs) // 3)
            rolling_avg = []
            for i in range(len(tracking_costs)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(tracking_costs), i + window_size // 2 + 1)
                rolling_avg.append(np.mean(tracking_costs[start_idx:end_idx]))
                
            ax3.plot(timesteps, tracking_costs, 'purple', alpha=0.5, linewidth=1, label='Raw')
            ax3.plot(timesteps, rolling_avg, 'darkviolet', linewidth=3, label=f'Rolling Avg (window={window_size})')
            ax3.set_xlabel('Episode Timestep')
            ax3.set_ylabel('Main MPC Tracking Cost')
            ax3.set_title('Tracking Cost with Rolling Average')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.plot(timesteps, tracking_costs, 'purple', linewidth=2, marker='o', markersize=4)
            ax3.set_xlabel('Episode Timestep')
            ax3.set_ylabel('Main MPC Tracking Cost')
            ax3.set_title('Tracking Cost (insufficient data for rolling avg)')
            ax3.grid(True, alpha=0.3)
        
        # Statistics summary
        ax4.axis('off')
        
        stats_text = f'Main MPC Tracking Cost Statistics\n\n'
        stats_text += f'Mean: {np.mean(tracking_costs):.6f}\n'
        stats_text += f'Std:  {np.std(tracking_costs):.6f}\n'
        stats_text += f'Min:  {np.min(tracking_costs):.6f}\n'
        stats_text += f'Max:  {np.max(tracking_costs):.6f}\n'
        stats_text += f'Total Data Points: {len(tracking_costs)}\n\n'
        
        stats_text += 'Percentiles:\n'
        stats_text += f'25th: {np.percentile(tracking_costs, 25):.6f}\n'
        stats_text += f'50th: {np.percentile(tracking_costs, 50):.6f}\n'
        stats_text += f'75th: {np.percentile(tracking_costs, 75):.6f}\n'
        stats_text += f'95th: {np.percentile(tracking_costs, 95):.6f}\n'
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle(f'Episode {self.episode_id} - Main MPC Tracking Cost Analysis', fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(self.episode_dir, 'main_mpc_tracking_costs.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved main MPC tracking costs analysis to {plot_path}') 


class ExperimentSummary:
    """Aggregate metrics across all episodes with enhanced trajectory analysis"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.episode_metrics = []
        
    def add_episode(self, episode_metrics):
        """Add metrics from a completed episode"""
        self.episode_metrics.append(episode_metrics)
        
    def save_summary(self):
        """Save experiment summary with aggregated metrics including trajectory comparisons"""
        if not self.episode_metrics:
            return
            
        # Calculate aggregated metrics
        successes = [ep['success'] for ep in self.episode_metrics]
        returns = [ep['episode_return'] for ep in self.episode_metrics]
        durations = [ep['episode_duration'] for ep in self.episode_metrics]
        position_mses = [ep['avg_position_rmse'] for ep in self.episode_metrics]
        orientation_distances = [ep['avg_orientation_distance'] for ep in self.episode_metrics]
        
        # NEW: Trajectory comparison metrics
        ref_vs_actual_pos = [ep.get('avg_ref_vs_actual_pos_rmse', 0.0) for ep in self.episode_metrics]
        ref_vs_mpc_pos = [ep.get('avg_ref_vs_mpc_pos_rmse', 0.0) for ep in self.episode_metrics]
        mpc_vs_actual_pos = [ep.get('avg_mpc_vs_actual_pos_rmse', 0.0) for ep in self.episode_metrics]
        ref_vs_actual_orient = [ep.get('avg_ref_vs_actual_orient_dist', 0.0) for ep in self.episode_metrics]
        ref_vs_mpc_orient = [ep.get('avg_ref_vs_mpc_orient_dist', 0.0) for ep in self.episode_metrics]
        mpc_vs_actual_orient = [ep.get('avg_mpc_vs_actual_orient_dist', 0.0) for ep in self.episode_metrics]
        
        # NEW: Main MPC tracking costs
        main_mpc_costs = [ep.get('avg_main_mpc_tracking_cost', 0.0) for ep in self.episode_metrics]
        
        # NEW: Vanilla trajectory comparison metrics
        vanilla_ref_vs_mpc_pos = [ep.get('avg_vanilla_ref_vs_mpc_pos_rmse', 0.0) for ep in self.episode_metrics]
        vanilla_ref_vs_mpc_orient = [ep.get('avg_vanilla_ref_vs_mpc_orient_dist', 0.0) for ep in self.episode_metrics]
        
        summary = {
            'experiment_summary': {
                'total_episodes': len(self.episode_metrics),
                'success_rate': float(np.mean(successes)),
                'avg_return': float(np.mean(returns)),
                'avg_episode_duration': float(np.mean(durations)),
                'avg_position_rmse': float(np.mean(position_mses)),
                'avg_orientation_distance': float(np.mean(orientation_distances)),
                'std_position_rmse': float(np.std(position_mses)),
                'std_orientation_distance': float(np.std(orientation_distances)),
                # NEW: Trajectory comparison metrics
                'avg_ref_vs_actual_pos_rmse': float(np.mean(ref_vs_actual_pos)),
                'avg_ref_vs_mpc_pos_rmse': float(np.mean(ref_vs_mpc_pos)),
                'avg_mpc_vs_actual_pos_rmse': float(np.mean(mpc_vs_actual_pos)),
                'avg_ref_vs_actual_orient_dist': float(np.mean(ref_vs_actual_orient)),
                'avg_ref_vs_mpc_orient_dist': float(np.mean(ref_vs_mpc_orient)),
                'avg_mpc_vs_actual_orient_dist': float(np.mean(mpc_vs_actual_orient)),
                'std_ref_vs_actual_pos_rmse': float(np.std(ref_vs_actual_pos)),
                'std_ref_vs_mpc_pos_rmse': float(np.std(ref_vs_mpc_pos)),
                'std_mpc_vs_actual_pos_rmse': float(np.std(mpc_vs_actual_pos)),
                'std_ref_vs_actual_orient_dist': float(np.std(ref_vs_actual_orient)),
                'std_ref_vs_mpc_orient_dist': float(np.std(ref_vs_mpc_orient)),
                'std_mpc_vs_actual_orient_dist': float(np.std(mpc_vs_actual_orient)),
                # NEW: Main MPC tracking costs
                'avg_main_mpc_tracking_cost': float(np.mean(main_mpc_costs)),
                'std_main_mpc_tracking_cost': float(np.std(main_mpc_costs)),
                # NEW: Vanilla trajectory comparison metrics
                'avg_vanilla_ref_vs_mpc_pos_rmse': float(np.mean(vanilla_ref_vs_mpc_pos)),
                'avg_vanilla_ref_vs_mpc_orient_dist': float(np.mean(vanilla_ref_vs_mpc_orient)),
                'std_vanilla_ref_vs_mpc_pos_rmse': float(np.std(vanilla_ref_vs_mpc_pos)),
                'std_vanilla_ref_vs_mpc_orient_dist': float(np.std(vanilla_ref_vs_mpc_orient))
            },
            'per_episode_metrics': self.episode_metrics
        }
        
        # Save to JSON
        summary_path = os.path.join(self.output_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save to CSV for easy analysis
        self.save_csv()
        
        # Create summary plots
        self.create_summary_plots()
        
        # Enhanced text summary
        self.save_text_summary(summary)
        
        return summary
        
    def save_csv(self):
        """Save episode metrics in CSV format with trajectory comparison data"""
        import csv
        
        csv_path = os.path.join(self.output_dir, 'episode_metrics.csv')
        
        fieldnames = ['episode_id', 'success', 'crashed', 'episode_return', 'highest_reward', 
                     'episode_duration', 'avg_position_rmse', 'avg_orientation_distance',
                     'avg_ref_vs_actual_pos_rmse', 'avg_ref_vs_mpc_pos_rmse', 'avg_mpc_vs_actual_pos_rmse',
                     'avg_ref_vs_actual_orient_dist', 'avg_ref_vs_mpc_orient_dist', 'avg_mpc_vs_actual_orient_dist',
                     'avg_vanilla_ref_vs_mpc_pos_rmse', 'avg_vanilla_ref_vs_mpc_orient_dist',
                     'avg_main_mpc_tracking_cost']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for ep in self.episode_metrics:
                row = {field: ep.get(field, 0.0) for field in fieldnames}
                writer.writerow(row)
                
    def create_summary_plots(self):
        """Create comprehensive summary visualization plots"""
        if len(self.episode_metrics) < 2:
            return
            
        # Extract data
        episode_ids = [ep['episode_id'] for ep in self.episode_metrics]
        successes = [ep['success'] for ep in self.episode_metrics]
        position_mses = [ep['avg_position_rmse'] for ep in self.episode_metrics]
        orientation_distances = [ep['avg_orientation_distance'] for ep in self.episode_metrics]
        returns = [ep['episode_return'] for ep in self.episode_metrics]
        
        # Trajectory comparison data
        ref_vs_actual_pos = [ep.get('avg_ref_vs_actual_pos_rmse', 0.0) for ep in self.episode_metrics]
        ref_vs_mpc_pos = [ep.get('avg_ref_vs_mpc_pos_rmse', 0.0) for ep in self.episode_metrics]
        mpc_vs_actual_pos = [ep.get('avg_mpc_vs_actual_pos_rmse', 0.0) for ep in self.episode_metrics]
        main_mpc_costs = [ep.get('avg_main_mpc_tracking_cost', 0.0) for ep in self.episode_metrics]
        
        # Create 2x3 subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Success rate over episodes
        success_colors = ['green' if s else 'red' for s in successes]
        axes[0,0].scatter(episode_ids, successes, c=success_colors, s=50, alpha=0.7)
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Success')
        axes[0,0].set_title(f'Success Rate: {np.mean(successes):.2%}')
        axes[0,0].set_ylim(-0.1, 1.1)
        axes[0,0].grid(True, alpha=0.3)
        
        # Position RMSE distribution
        axes[0,1].hist(position_mses, bins=min(10, len(position_mses)), alpha=0.7, color='blue', edgecolor='black')
        axes[0,1].axvline(np.mean(position_mses), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(position_mses):.6f}')
        axes[0,1].set_xlabel('Average Position RMSE (m)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Position RMSE Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Orientation distance distribution
        axes[0,2].hist(orientation_distances, bins=min(10, len(orientation_distances)), alpha=0.7, color='orange', edgecolor='black')
        axes[0,2].axvline(np.mean(orientation_distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(orientation_distances):.6f}')
        axes[0,2].set_xlabel('Average Orientation Distance (rad)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Orientation Distance Distribution')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # Trajectory comparison over episodes
        axes[1,0].plot(episode_ids, ref_vs_actual_pos, 'b-o', linewidth=2, markersize=4, alpha=0.8, label='Ref vs Actual')
        axes[1,0].plot(episode_ids, ref_vs_mpc_pos, 'r-s', linewidth=2, markersize=4, alpha=0.8, label='Ref vs MPC')
        axes[1,0].plot(episode_ids, mpc_vs_actual_pos, 'g-^', linewidth=2, markersize=4, alpha=0.8, label='MPC vs Actual')
        axes[1,0].set_xlabel('Episode')
        axes[1,0].set_ylabel('Position RMSE (m)')
        axes[1,0].set_title('Trajectory Position Errors Across Episodes')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Main MPC tracking costs
        axes[1,1].plot(episode_ids, main_mpc_costs, 'purple', linewidth=2, marker='o', markersize=4, alpha=0.8)
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Main MPC Tracking Cost')
        axes[1,1].set_title(f'Main MPC Tracking Cost Across Episodes\nAvg: {np.mean(main_mpc_costs):.6f}')
        axes[1,1].grid(True, alpha=0.3)
        
        # Success vs tracking accuracy
        axes[1,2].scatter([position_mses[i] for i, s in enumerate(successes) if s], 
                   [orientation_distances[i] for i, s in enumerate(successes) if s], 
                         c='green', label='Success', s=50, alpha=0.7)
        axes[1,2].scatter([position_mses[i] for i, s in enumerate(successes) if not s], 
                   [orientation_distances[i] for i, s in enumerate(successes) if not s], 
                         c='red', label='Failure', s=50, alpha=0.7)
        axes[1,2].set_xlabel('Average Position RMSE (m)')
        axes[1,2].set_ylabel('Average Orientation Distance (rad)')
        axes[1,2].set_title('Success vs Tracking Accuracy')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'experiment_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_text_summary(self, summary):
        """Save enhanced text summary with trajectory comparison metrics"""
        exp_summary = summary['experiment_summary']
        
        summary_text = f"""
EXPERIMENT SUMMARY
==================

Total Episodes: {exp_summary['total_episodes']}
Success Rate: {exp_summary['success_rate']:.2%}
Average Return: {exp_summary['avg_return']:.3f}
Average Episode Duration: {exp_summary['avg_episode_duration']:.2f}s

TRACKING ACCURACY
=================
Average Position RMSE: {exp_summary['avg_position_rmse']:.6f} ± {exp_summary['std_position_rmse']:.6f} m
Average Orientation Distance: {exp_summary['avg_orientation_distance']:.6f} ± {exp_summary['std_orientation_distance']:.6f} rad

MPC TRACKING COST
=================
Average Main MPC Tracking Cost: {exp_summary['avg_main_mpc_tracking_cost']:.6f} ± {exp_summary['std_main_mpc_tracking_cost']:.6f}

TRAJECTORY COMPARISON ANALYSIS
==============================
Reference vs Actual Position RMSE: {exp_summary['avg_ref_vs_actual_pos_rmse']:.6f} ± {exp_summary['std_ref_vs_actual_pos_rmse']:.6f} m
Reference vs MPC Position RMSE: {exp_summary['avg_ref_vs_mpc_pos_rmse']:.6f} ± {exp_summary['std_ref_vs_mpc_pos_rmse']:.6f} m
MPC vs Actual Position RMSE: {exp_summary['avg_mpc_vs_actual_pos_rmse']:.6f} ± {exp_summary['std_mpc_vs_actual_pos_rmse']:.6f} m

Reference vs Actual Orientation Distance: {exp_summary['avg_ref_vs_actual_orient_dist']:.6f} ± {exp_summary['std_ref_vs_actual_orient_dist']:.6f} rad
Reference vs MPC Orientation Distance: {exp_summary['avg_ref_vs_mpc_orient_dist']:.6f} ± {exp_summary['std_ref_vs_mpc_orient_dist']:.6f} rad
MPC vs Actual Orientation Distance: {exp_summary['avg_mpc_vs_actual_orient_dist']:.6f} ± {exp_summary['std_mpc_vs_actual_orient_dist']:.6f} rad

REWARD DISTRIBUTION
===================
"""
        
        # Add reward distribution
        max_reward = max([ep['highest_reward'] for ep in self.episode_metrics])
        for r in range(int(max_reward) + 1):
            count = sum(1 for ep in self.episode_metrics if ep['highest_reward'] >= r)
            rate = count / len(self.episode_metrics)
            summary_text += f"Reward >= {r}: {count}/{len(self.episode_metrics)} = {rate:.1%}\n"
        
        summary_text += f"\nPER-EPISODE RETURNS\n{[ep['episode_return'] for ep in self.episode_metrics]}\n"
        summary_text += f"\nPER-EPISODE HIGHEST REWARDS\n{[ep['highest_reward'] for ep in self.episode_metrics]}\n"
        summary_text += f"\nPER-EPISODE MAIN MPC TRACKING COSTS\n{[round(ep.get('avg_main_mpc_tracking_cost', 0.0), 6) for ep in self.episode_metrics]}\n"
        
        # Save to file
        with open(os.path.join(self.output_dir, 'experiment_summary.txt'), 'w') as f:
            f.write(summary_text)


def save_diffusion_analysis(policy, episode_tracker, inference_timestep, actual_trajectory_segment, actual_quaternion_segment=None):
    """
    Save simplified diffusion trajectory analysis for a specific inference step
    
    Args:
        policy: The diffusion policy with last_diffusion_data
        episode_tracker: EpisodeMetricsTracker instance
        inference_timestep: The environment timestep when inference occurred
        actual_trajectory_segment: List of 32 actual end-effector positions executed
        actual_quaternion_segment: List of 32 actual quaternion segments executed (optional)
    """
    if not hasattr(policy, 'last_diffusion_data') or policy.last_diffusion_data is None:
        return
        
    # Create inference directory structure
    inference_dir = os.path.join(episode_tracker.episode_dir, 'inference', f'step_{inference_timestep:04d}')
    os.makedirs(inference_dir, exist_ok=True)
    
    # Extract minimal diffusion data
    diffusion_data = policy.last_diffusion_data
    
    # Get final trajectory (H, 8) - [x, y, z, w, x, y, z, gripper]
    final_traj = diffusion_data.get('final_trajectory', None)
    
    if final_traj is None:
        return
    
    # Reference trajectory is the final trajectory from policy
    reference_traj = final_traj  # (8, 8) or (H, 8)
    
    # Process actual trajectory with robust error handling
    if actual_trajectory_segment is not None:
        try:
            actual_traj_raw = np.array(actual_trajectory_segment)
            
            # Handle different input formats
            if actual_traj_raw.ndim == 2:
                if actual_traj_raw.shape[1] >= 7:  # Has position + quaternion + gripper
                    actual_traj = actual_traj_raw[:, :8]  # Take first 8 columns
                elif actual_traj_raw.shape[1] == 3:  # Only positions
                    actual_traj = np.zeros((actual_traj_raw.shape[0], 8))
                    actual_traj[:, :3] = actual_traj_raw  # positions
                    actual_traj[:, 3:7] = np.tile([1, 0, 0, 0], (actual_traj_raw.shape[0], 1))  # default quaternions
                    actual_traj[:, 7] = 0.5  # default gripper
                else:
                    actual_traj = np.zeros((32, 8))
            elif actual_traj_raw.ndim == 1:
                actual_traj = np.zeros((32, 8))
            else:
                actual_traj = np.zeros((32, 8))
                
        except Exception as e:
            actual_traj = np.zeros((32, 8))
    else:
        actual_traj = np.zeros((32, 8))
    
    # Only proceed with trajectory comparison if we have valid data
    if actual_traj.shape[1] >= 7 and reference_traj.shape[1] >= 7:
        # Add trajectory comparison data to episode tracker
        episode_tracker.add_trajectory_comparison(
            timestep=inference_timestep,
            reference_traj=reference_traj,  # (8, 8)
            actual_traj=actual_traj         # (32, 8)
        )
        
        # Create trajectory comparison plot
        episode_tracker.plot_trajectory_comparison_analysis(
            timestep=inference_timestep,
            reference_traj=reference_traj,
            actual_traj=actual_traj,
            inference_dir=inference_dir,
            vanilla_reference_traj=None  # No vanilla trajectories in minimal logging
        )
    
    # Save minimal diffusion data to JSON
    diffusion_data_path = os.path.join(inference_dir, 'diffusion_data.json')
    
    # Function to convert numpy arrays to lists recursively
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {
        'trajectories': [traj.tolist() for traj in diffusion_data.get('trajectories', [])],
        'trajectories_post_grad': [traj.tolist() for traj in diffusion_data.get('trajectories_post_grad', [])],
        'trajectories_guidance': [traj.tolist() for traj in diffusion_data.get('trajectories_guidance', [])],
        'trajectories_noisy': [traj.tolist() for traj in diffusion_data.get('trajectories_noisy', [])],
        'quaternions': [quat.tolist() for quat in diffusion_data.get('quaternions', [])],
        'quaternions_post_grad': [quat.tolist() for quat in diffusion_data.get('quaternions_post_grad', [])],
        'quaternions_guidance': [quat.tolist() for quat in diffusion_data.get('quaternions_guidance', [])],
        'quaternions_noisy': [quat.tolist() for quat in diffusion_data.get('quaternions_noisy', [])],
        'costs': convert_numpy_to_list(diffusion_data.get('costs', [])),
        'timesteps': diffusion_data.get('timesteps', []),
        'inference_env_timestep': inference_timestep,
        'actual_trajectory': [pos.tolist() if hasattr(pos, 'tolist') else list(pos) for pos in actual_trajectory_segment] if actual_trajectory_segment else None,
        'actual_quaternion': [quat.tolist() if hasattr(quat, 'tolist') else list(quat) for quat in actual_quaternion_segment] if actual_quaternion_segment else None,
        # Vanilla diffusion data
        'vanilla_trajectory': diffusion_data.get('vanilla_trajectory', None).tolist() if diffusion_data.get('vanilla_trajectory', None) is not None else None,
        'vanilla_mpc_prediction': convert_numpy_to_list(diffusion_data.get('vanilla_mpc_prediction', None))
    }
    
    with open(diffusion_data_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Saved diffusion analysis to {inference_dir}")