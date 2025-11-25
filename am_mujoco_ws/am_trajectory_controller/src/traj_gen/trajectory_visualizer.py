#!/usr/bin/env python3
"""
Trajectory Visualizer and Editor for Aerial Manipulator
Provides tools to visualize, analyze, and modify end-effector trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from scipy.spatial.transform import Rotation as R
from traj_util import slerpn, vis_traj

class TrajectoryVisualizer:
    def __init__(self, csv_path):
        """
        Initialize with trajectory CSV file.
        Expected format: t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,joint3,joint4,gripper
        """
        self.csv_path = csv_path
        self.data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        self.parse_trajectory()
        
    def parse_trajectory(self):
        """Parse the loaded CSV data into meaningful components."""
        self.time = self.data[:, 0]
        self.position = self.data[:, 1:4]  # px, py, pz
        self.quaternion = self.data[:, 4:8]  # qx, qy, qz, qw (XYZW format)
        self.velocity = self.data[:, 8:11]  # vx, vy, vz
        self.control_input = self.data[:, 11:14]  # ux, uy, uz
        self.joint_angles = self.data[:, 14:18]  # joint1-4
        self.gripper = self.data[:, 18]
        
        # Convert quaternions to Euler angles for easier interpretation
        self.euler_angles = np.array([
            R.from_quat(quat).as_euler('xyz', degrees=True) 
            for quat in self.quaternion
        ])
        
        self.dt = self.time[1] - self.time[0] if len(self.time) > 1 else 0.01
        self.duration = self.time[-1] - self.time[0]
        
    def print_stats(self):
        """Print trajectory statistics."""
        print(f"\n=== Trajectory Statistics ===")
        print(f"Duration: {self.duration:.2f} seconds")
        print(f"Time step: {self.dt:.4f} seconds")
        print(f"Number of points: {len(self.time)}")
        
        print(f"\n--- Position Statistics ---")
        pos_min = np.min(self.position, axis=0)
        pos_max = np.max(self.position, axis=0)
        pos_range = pos_max - pos_min
        print(f"X: [{pos_min[0]:.3f}, {pos_max[0]:.3f}] range: {pos_range[0]:.3f}m")
        print(f"Y: [{pos_min[1]:.3f}, {pos_max[1]:.3f}] range: {pos_range[1]:.3f}m")
        print(f"Z: [{pos_min[2]:.3f}, {pos_max[2]:.3f}] range: {pos_range[2]:.3f}m")
        
        print(f"\n--- Velocity Statistics ---")
        vel_max = np.max(np.abs(self.velocity), axis=0)
        vel_mean = np.mean(np.abs(self.velocity), axis=0)
        print(f"Max velocities: [{vel_max[0]:.4f}, {vel_max[1]:.4f}, {vel_max[2]:.4f}] m/s")
        print(f"Mean velocities: [{vel_mean[0]:.4f}, {vel_mean[1]:.4f}, {vel_mean[2]:.4f}] m/s")
        
        print(f"\n--- Orientation Statistics ---")
        euler_min = np.min(self.euler_angles, axis=0)
        euler_max = np.max(self.euler_angles, axis=0)
        euler_range = euler_max - euler_min
        axis_labels = ['Roll', 'Pitch', 'Yaw']
        for idx, label in enumerate(axis_labels):
            print(f"{label}: [{euler_min[idx]:.2f}°, {euler_max[idx]:.2f}°] range: {euler_range[idx]:.2f}°")
        
    def plot_3d_trajectory(self, save_path=None, show_orientation=False, subsample=10):
        """Plot 3D trajectory with optional orientation arrows."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory path
        ax.plot(self.position[:, 0], self.position[:, 1], self.position[:, 2], 
                'b-', linewidth=2, label='End-Effector Path')
        
        # Mark start and end points
        ax.scatter(*self.position[0], color='green', s=100, label='Start')
        ax.scatter(*self.position[-1], color='red', s=100, label='End')
        
        # Show orientation arrows if requested
        if show_orientation:
            # Subsample for clarity
            indices = range(0, len(self.position), subsample)
            for i in indices:
                pos = self.position[i]
                # Convert quaternion to rotation matrix
                rot = R.from_quat(self.quaternion[i])
                # Get x-axis direction (forward direction of end-effector)
                direction = rot.apply([0.02, 0, 0])  # 2cm arrow length
                ax.quiver(pos[0], pos[1], pos[2], 
                         direction[0], direction[1], direction[2],
                         color='red', alpha=0.6, arrow_length_ratio=0.3)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D End-Effector Trajectory')
        ax.legend()
        ax.grid(True)
        
        # Equal aspect ratio
        max_range = np.array([
            self.position[:, 0].max() - self.position[:, 0].min(),
            self.position[:, 1].max() - self.position[:, 1].min(),
            self.position[:, 2].max() - self.position[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (self.position[:, 0].max() + self.position[:, 0].min()) * 0.5
        mid_y = (self.position[:, 1].max() + self.position[:, 1].min()) * 0.5
        mid_z = (self.position[:, 2].max() + self.position[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D trajectory plot saved to: {save_path}")
        plt.show()
        
    def plot_comprehensive_analysis(self, save_path=None):
        """Create comprehensive trajectory analysis plots."""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Position vs time
        ax1 = plt.subplot(3, 4, 1)
        for i, label in enumerate(['X', 'Y', 'Z']):
            plt.plot(self.time, self.position[:, i], label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('End-Effector Position vs Time')
        plt.legend()
        plt.grid(True)
        
        # 2. Velocity vs time
        ax2 = plt.subplot(3, 4, 2)
        for i, label in enumerate(['Vx', 'Vy', 'Vz']):
            plt.plot(self.time, self.velocity[:, i], label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('End-Effector Velocity vs Time')
        plt.legend()
        plt.grid(True)
        
        # 3. Euler angles vs time
        ax3 = plt.subplot(3, 4, 3)
        for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
            plt.plot(self.time, self.euler_angles[:, i], label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title('End-Effector Orientation vs Time')
        plt.legend()
        plt.grid(True)
        
        # 4. Control inputs vs time
        ax4 = plt.subplot(3, 4, 4)
        for i, label in enumerate(['Ux', 'Uy', 'Uz']):
            plt.plot(self.time, self.control_input[:, i], label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Inputs vs Time')
        plt.legend()
        plt.grid(True)
        
        # 5. Joint angles vs time
        ax5 = plt.subplot(3, 4, 5)
        for i, label in enumerate(['Joint1', 'Joint2', 'Joint3', 'Joint4']):
            plt.plot(self.time, self.joint_angles[:, i] * 180/np.pi, label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angle (deg)')
        plt.title('Arm Joint Angles vs Time')
        plt.legend()
        plt.grid(True)
        
        # 6. Speed profile
        ax6 = plt.subplot(3, 4, 6)
        speed = np.linalg.norm(self.velocity, axis=1)
        plt.plot(self.time, speed)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.title('End-Effector Speed Profile')
        plt.grid(True)
        
        # 7. XY trajectory
        ax7 = plt.subplot(3, 4, 7)
        plt.plot(self.position[:, 0], self.position[:, 1], 'b-', linewidth=2)
        plt.scatter(self.position[0, 0], self.position[0, 1], color='green', s=100, label='Start')
        plt.scatter(self.position[-1, 0], self.position[-1, 1], color='red', s=100, label='End')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('XY Plane Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 8. XZ trajectory
        ax8 = plt.subplot(3, 4, 8)
        plt.plot(self.position[:, 0], self.position[:, 2], 'b-', linewidth=2)
        plt.scatter(self.position[0, 0], self.position[0, 2], color='green', s=100, label='Start')
        plt.scatter(self.position[-1, 0], self.position[-1, 2], color='red', s=100, label='End')
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.title('XZ Plane Trajectory')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # 9. Acceleration profile
        ax9 = plt.subplot(3, 4, 9)
        if len(self.velocity) > 1:
            acceleration = np.diff(self.velocity, axis=0) / self.dt
            for i, label in enumerate(['Ax', 'Ay', 'Az']):
                plt.plot(self.time[1:], acceleration[:, i], label=f'{label}')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (m/s²)')
            plt.title('End-Effector Acceleration')
            plt.legend()
            plt.grid(True)
        
        # 10. Gripper state
        ax10 = plt.subplot(3, 4, 10)
        plt.plot(self.time, self.gripper)
        plt.xlabel('Time (s)')
        plt.ylabel('Gripper State')
        plt.title('Gripper State vs Time')
        plt.grid(True)
        
        # 11. Position error from start
        ax11 = plt.subplot(3, 4, 11)
        position_error = np.linalg.norm(self.position - self.position[0], axis=1)
        plt.plot(self.time, position_error)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance from Start (m)')
        plt.title('Cumulative Position Change')
        plt.grid(True)
        
        # 12. Quaternion components
        ax12 = plt.subplot(3, 4, 12)
        for i, label in enumerate(['qx', 'qy', 'qz', 'qw']):
            plt.plot(self.time, self.quaternion[:, i], label=f'{label}')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion Component')
        plt.title('Quaternion Components vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive analysis plot saved to: {save_path}")
        plt.show()

class TrajectoryModifier:
    def __init__(self, visualizer):
        self.vis = visualizer
        
    def create_circle_trajectory(self, center=[0.3, 0.0, 0.3], radius=0.05, 
                                duration=10.0, dt=0.01, orientation='constant'):
        """Create a circular trajectory."""
        t = np.arange(0, duration, dt)
        omega = 2 * np.pi / duration  # One full circle
        
        # Circular motion in XY plane
        x = center[0] + radius * np.cos(omega * t)
        y = center[1] + radius * np.sin(omega * t)
        z = np.full_like(t, center[2])
        
        position = np.column_stack([x, y, z])
        
        # Velocity (analytical derivative)
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        vz = np.zeros_like(t)
        velocity = np.column_stack([vx, vy, vz])
        
        # Orientation options
        if orientation == 'constant':
            # Keep orientation constant
            quaternion = np.tile([0, 0, 0, 1], (len(t), 1))  # Identity quaternion
        elif orientation == 'tangent':
            # Orient tangent to circle (facing motion direction)
            angles = omega * t + np.pi/2  # Tangent direction
            quaternion = np.array([
                R.from_euler('z', angle).as_quat() for angle in angles
            ])
        elif orientation == 'inward':
            # Orient toward center of circle
            angles = omega * t + np.pi  # Inward direction
            quaternion = np.array([
                R.from_euler('z', angle).as_quat() for angle in angles
            ])
        
        return self._create_full_trajectory(t, position, quaternion, velocity)
    
    def create_figure_eight(self, center=[0.3, 0.0, 0.3], radius=0.05, 
                           duration=15.0, dt=0.01):
        """Create a figure-eight trajectory."""
        t = np.arange(0, duration, dt)
        omega = 2 * np.pi / duration
        
        # Figure-eight (Lissajous curve with 2:1 frequency ratio)
        x = center[0] + radius * np.cos(omega * t)
        y = center[1] + radius * np.sin(2 * omega * t) / 2
        z = np.full_like(t, center[2])
        
        position = np.column_stack([x, y, z])
        
        # Velocity
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(2 * omega * t)
        vz = np.zeros_like(t)
        velocity = np.column_stack([vx, vy, vz])
        
        # Constant orientation
        quaternion = np.tile([0, 0, 0, 1], (len(t), 1))
        
        return self._create_full_trajectory(t, position, quaternion, velocity)
    
    def create_spiral_trajectory(self, output_dir=".", speed_factor=1.0):
        """Create spiral trajectory with adjustable speed."""
        # Adjusted duration based on speed factor (slower = longer duration)
        base_duration = 12.0
        duration = base_duration / speed_factor
        
        t = np.arange(0, duration, 0.01)
        # Start from the ACTUAL end-effector position in the simulation (not CSV reference)
        center = np.array([0.22, -0.02, 0.38])  # Actual EE starting position from forward kinematics
        
        # Spiral parameters with MUCH larger movements (0.5m range in each direction)
        max_radius = 0.5  # 1.0m diameter in XY plane
        z_range = 1.0      # 1.0m vertical movement
        
        # Scale movements by speed factor
        radius_growth = max_radius / duration  # Grows to max radius over time
        height_change = z_range / duration     # Total Z movement over duration
        angular_freq = 2 * np.pi / (duration/4)  # Complete 4 turns over duration
        
        # Scale angular frequency by speed factor
        angular_freq *= speed_factor
        
        # Generate spiral with progressive radius growth
        radius = radius_growth * t
        angle = angular_freq * t
        
        # Large movements in X and Y (spiral pattern)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        
        # Large Z movement with some variation for interest
        # Start low, go high, then come back down
        z_progress = t / duration
        z = center[2] + z_range * (0.5 * np.sin(2 * np.pi * z_progress) + 0.5 * z_progress - 0.25)
        
        position = np.column_stack([x, y, z])
        
        # Calculate velocity
        vx = np.gradient(x, 0.01)
        vy = np.gradient(y, 0.01)
        vz = np.gradient(z, 0.01)
        velocity = np.column_stack([vx, vy, vz])
        
        # Calculate control inputs (simplified)
        control = np.zeros_like(velocity)
        
        # Create joint angles (keep simple for now)
        joint_angles = np.tile([10.0, 12.0, 90.0, 0.0], (len(t), 1)) * np.pi/180
        
        # Constant orientation
        quaternion = np.tile([0, 0, 0, 1], (len(t), 1))
        
        # Create full trajectory
        traj_data = np.column_stack([
            t, position, quaternion, velocity, control, joint_angles, np.zeros(len(t))
        ])
        
        # Save trajectory
        output_file = os.path.join(output_dir, "spiral_trajectory.csv")
        header = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,joint3,joint4,gripper"
        np.savetxt(output_file, traj_data, delimiter=',', header=header, comments='')
        print(f"Trajectory saved to: {output_file}")
        print(f"Starting from ACTUAL EE position: {center}")
        print(f"Movement ranges - X: {max_radius*2:.1f}m, Y: {max_radius*2:.1f}m, Z: {z_range:.1f}m")
        
        # Create visualization
        temp_viz = TrajectoryVisualizer(output_file)
        temp_viz.print_stats()
        plot_path = os.path.join(output_dir, "spiral_3d_plot.png")
        temp_viz.plot_3d_trajectory(save_path=plot_path, show_orientation=True)
        
        return traj_data
    
    def create_spiral_angle_trajectory(self, output_dir=".", speed_factor=1.0):
        """Create spiral trajectory with adjustable speed AND dynamic orientation changes."""
        # Fixed duration of 60 seconds as requested
        duration = 60.0
        
        t = np.arange(0, duration, 0.01)
        # Start from the ACTUAL end-effector position in the simulation (not CSV reference)
        center = np.array([0.22, -0.02, 0.38])  # Actual EE starting position from forward kinematics
        
        # Spiral parameters with MUCH larger movements (1.0m range in each direction)
        max_radius = 0.5  # 1.0m diameter in XY plane
        z_range = 1.0      # 1.0m vertical movement
        
        # Scale movements by speed factor for the motion characteristics
        radius_growth = max_radius / duration  # Grows to max radius over time
        height_change = z_range / duration     # Total Z movement over duration
        angular_freq = 2 * np.pi / (duration/4)  # Complete 4 turns over duration
        
        # Scale angular frequency by speed factor
        angular_freq *= speed_factor
        
        # Generate spiral with progressive radius growth
        radius = radius_growth * t
        angle = angular_freq * t
        
        # Large movements in X and Y (spiral pattern)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        
        # Large Z movement with some variation for interest
        # Start low, go high, then come back down
        z_progress = t / duration
        z = center[2] + z_range * (0.5 * np.sin(2 * np.pi * z_progress) + 0.5 * z_progress - 0.25)
        
        position = np.column_stack([x, y, z])
        
        # Calculate velocity
        vx = np.gradient(x, 0.01)
        vy = np.gradient(y, 0.01)
        vz = np.gradient(z, 0.01)
        velocity = np.column_stack([vx, vy, vz])
        
        # Calculate control inputs (simplified)
        control = np.zeros_like(velocity)
        
        # Create joint angles (keep simple for now)
        joint_angles = np.tile([10.0, 12.0, 90.0, 0.0], (len(t), 1)) * np.pi/180
        
        # DYNAMIC ORIENTATION: 0° to 15° and back to 0° for roll, pitch, yaw
        max_angle = 15.0 * np.pi/180  # 15 degrees in radians
        
        # All axes follow the same single cycle pattern: 0° -> 15° -> 0°
        # sin(π*t/T) goes 0 -> 1 -> 0 over duration T
        angle_pattern = max_angle * np.sin(np.pi * t / duration)
        
        # Same pattern for all three axes
        roll = angle_pattern
        pitch = angle_pattern  
        yaw = angle_pattern
        
        # Convert RPY to quaternions
        euler_angles = np.column_stack([roll, pitch, yaw])
        quaternions = []
        
        for rpy in euler_angles:
            rot = R.from_euler('xyz', rpy)
            quat_xyzw = rot.as_quat()  # Returns [x, y, z, w]
            quaternions.append(quat_xyzw)
        
        quaternion = np.array(quaternions)
        
        # Create full trajectory
        traj_data = np.column_stack([
            t, position, quaternion, velocity, control, joint_angles, np.zeros(len(t))
        ])
        
        # Save trajectory
        output_file = os.path.join(output_dir, "spiral_angle_trajectory.csv")
        header = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,joint3,joint4,gripper"
        np.savetxt(output_file, traj_data, delimiter=',', header=header, comments='')
        print(f"Trajectory saved to: {output_file}")
        print(f"Duration: {duration} seconds")
        print(f"Starting from ACTUAL EE position: {center}")
        print(f"Movement ranges - X: {max_radius*2:.1f}m, Y: {max_radius*2:.1f}m, Z: {z_range:.1f}m")
        print(f"Orientation ranges - Roll: 0-15°, Pitch: 0-15°, Yaw: 0-15° (all single cycle)")
        
        # Create visualization
        temp_viz = TrajectoryVisualizer(output_file)
        temp_viz.print_stats()
        plot_path = os.path.join(output_dir, "spiral_angle_3d_plot.png")
        temp_viz.plot_3d_trajectory(save_path=plot_path, show_orientation=True)
        
        return traj_data
    
    def create_zigzag_trajectory(self, start=[0.25, -0.05, 0.3], end=[0.35, 0.05, 0.3], 
                                zigzag_amplitude=0.02, num_cycles=5, duration=8.0, dt=0.01):
        """Create a zigzag trajectory."""
        t = np.arange(0, duration, dt)
        
        # Linear progression from start to end
        progress = t / duration
        x = start[0] + (end[0] - start[0]) * progress
        z = start[2] + (end[2] - start[2]) * progress
        
        # Zigzag in Y direction
        y = start[1] + (end[1] - start[1]) * progress + \
            zigzag_amplitude * np.sin(2 * np.pi * num_cycles * progress)
        
        position = np.column_stack([x, y, z])
        velocity = np.gradient(position, dt, axis=0)
        
        # Constant orientation
        quaternion = np.tile([0, 0, 0, 1], (len(t), 1))
        
        return self._create_full_trajectory(t, position, quaternion, velocity)
    
    def _create_full_trajectory(self, time, position, quaternion, velocity):
        """Create full trajectory data structure."""
        # Control inputs (zeros for now)
        control_input = np.zeros_like(velocity)
        
        # Joint angles (zeros for now)
        joint_angles = np.zeros((len(time), 4))
        
        # Gripper state (closed = 0)
        gripper = np.zeros(len(time))
        
        # Combine all data
        full_data = np.column_stack([
            time,
            position,          # px, py, pz
            quaternion,        # qx, qy, qz, qw
            velocity,          # vx, vy, vz
            control_input,     # ux, uy, uz
            joint_angles,      # joint1, joint2, joint3, joint4
            gripper           # gripper
        ])
        
        return full_data
    
    def save_trajectory(self, trajectory_data, output_path):
        """Save trajectory to CSV file."""
        header = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,joint3,joint4,gripper"
        np.savetxt(output_path, trajectory_data, delimiter=',', 
                   header=header, comments='', fmt='%.6f')
        print(f"Trajectory saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Trajectory Visualizer and Editor')
    parser.add_argument('trajectory', help='Path to trajectory CSV file')
    parser.add_argument('--action', choices=['visualize', 'analyze', 'create'], 
                       default='visualize', help='Action to perform')
    parser.add_argument('--output', default='.', help='Output directory for plots and files')
    parser.add_argument('--type', choices=['circle', 'figure8', 'spiral', 'zigzag', 'spiral_angle'], 
                       default='circle', help='Type of trajectory to create')
    parser.add_argument('--speed_factor', type=float, default=1.0, 
                       help='Speed factor for trajectory (1.0=normal, 0.5=half speed, 2.0=double speed)')
    
    args = parser.parse_args()
    
    viz = TrajectoryVisualizer(args.trajectory)
    
    if args.action == 'visualize':
        viz.plot_3d_trajectory(save_path=os.path.join(args.output, "3d_trajectory.png"), show_orientation=True)
    elif args.action == 'analyze':
        viz.print_stats()
        viz.plot_comprehensive_analysis(save_path=os.path.join(args.output, "trajectory_analysis.png"))
    elif args.action == 'create':
        # Create modifier for trajectory generation
        modifier = TrajectoryModifier(viz)
        
        if args.type == 'circle':
            traj_data = modifier.create_circle_trajectory()
            output_file = os.path.join(args.output, "circle_trajectory.csv")
        elif args.type == 'figure8':
            traj_data = modifier.create_figure_eight()
            output_file = os.path.join(args.output, "figure8_trajectory.csv")
        elif args.type == 'spiral':
            # Pass speed_factor to spiral creation
            modifier.create_spiral_trajectory(args.output, speed_factor=args.speed_factor)
            return  # This method handles its own saving and visualization
        elif args.type == 'zigzag':
            traj_data = modifier.create_zigzag_trajectory()
            output_file = os.path.join(args.output, "zigzag_trajectory.csv")
        elif args.type == 'spiral_angle':
            # Pass speed_factor to spiral_angle creation
            modifier.create_spiral_angle_trajectory(args.output, speed_factor=args.speed_factor)
            return  # This method handles its own saving and visualization
        
        # Save and visualize for non-spiral trajectories
        if args.type != 'spiral' and args.type != 'spiral_angle':
            modifier.save_trajectory(traj_data, output_file)
            
            # Create visualization
            new_viz = TrajectoryVisualizer(output_file)
            new_viz.print_stats()
            plot_path = os.path.join(args.output, f"{args.type}_3d_plot.png")
            new_viz.plot_3d_trajectory(save_path=plot_path, show_orientation=True)

if __name__ == "__main__":
    main() 