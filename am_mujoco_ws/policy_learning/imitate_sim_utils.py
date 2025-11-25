import torch
import numpy as np
import cv2
import mujoco
from einops import rearrange
from scipy.spatial.transform import Rotation as R, Slerp

from constants import DT


def quat_wxyz_to_axis_angle(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to axis-angle representation."""
    if np.linalg.norm(quat_wxyz) < 1e-6:
        return np.zeros(3)
    
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw).as_rotvec()


def apply_mpc_coordinate_transform(trajectory: np.ndarray) -> np.ndarray:
    """Apply -90° rotation about X axis to MPC trajectory quaternions."""
    coord_frame_transform = R.from_rotvec([-np.pi / 2, 0.0, 0.0])
    trajectory = trajectory.copy()
    
    for i in range(trajectory.shape[0]):
        quat_wxyz = trajectory[i, 3:7]
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        new_quat_xyzw = (coord_frame_transform * R.from_quat(quat_xyzw)).as_quat()
        trajectory[i, 3:7] = np.array([new_quat_xyzw[3], new_quat_xyzw[0], 
                                       -new_quat_xyzw[2], new_quat_xyzw[1]])
    return trajectory


def extract_robot_state(obs: dict) -> tuple:
    """Extract robot state from observation. Returns (qpos_8d, pos, quat_wxyz, gripper)."""
    qpos_8d = np.array(obs['qpos'])
    robot_pos = qpos_8d[:3]
    robot_quat_wxyz = qpos_8d[3:7]
    robot_gripper = np.clip(qpos_8d[7:8], 0.0, 1.0)
    return qpos_8d, robot_pos, robot_quat_wxyz, robot_gripper


def prepare_umi_image_observation(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, num_cameras, C, H, W) to (H, W, C) numpy array."""
    img_tensor = image_tensor.squeeze(0)[0]
    img_numpy = img_tensor.cpu().numpy()
    return img_numpy.transpose(1, 2, 0)


class ObservationBufferManager:
    """Manage sliding window observation buffer for UMI policy."""
    
    def __init__(self, img_obs_horizon: int, low_dim_obs_horizon: int):
        self.img_obs_horizon = img_obs_horizon
        self.low_dim_obs_horizon = low_dim_obs_horizon
        self.buffer = {
            'camera0_rgb': [],
            'robot0_eef_pos': [],
            'robot0_eef_rot_axis_angle': [],
            'robot0_gripper_width': []
        }
        self.low_dim_keys = ['robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width']
        self.initialized = False
    
    def add_observation(self, img_hwc: np.ndarray, robot_pos: np.ndarray, 
                       robot_rot_axis_angle: np.ndarray, robot_gripper: np.ndarray):
        """Add new observation to buffer with sliding window."""
        env_obs_t = {
            'camera0_rgb': img_hwc[np.newaxis, ...],
            'robot0_eef_pos': robot_pos[np.newaxis, :],
            'robot0_eef_rot_axis_angle': robot_rot_axis_angle[np.newaxis, :],
            'robot0_gripper_width': robot_gripper[np.newaxis, :],
        }
        
        for key in self.buffer:
            self.buffer[key].append(env_obs_t[key])
        
        if len(self.buffer['camera0_rgb']) > self.img_obs_horizon:
            self.buffer['camera0_rgb'] = self.buffer['camera0_rgb'][-self.img_obs_horizon:]
        
        for key in self.low_dim_keys:
            if len(self.buffer[key]) > self.low_dim_obs_horizon:
                self.buffer[key] = self.buffer[key][-self.low_dim_obs_horizon:]
        
        if not self.initialized:
            while len(self.buffer['camera0_rgb']) < self.img_obs_horizon:
                self.buffer['camera0_rgb'].insert(0, env_obs_t['camera0_rgb'].copy())
            
            for key in self.low_dim_keys:
                while len(self.buffer[key]) < self.low_dim_obs_horizon:
                    self.buffer[key].insert(0, env_obs_t[key].copy())
            
            self.initialized = True
    
    def get_stacked_observation(self) -> dict:
        """Get stacked observation for UMI policy input."""
        return {
            'camera0_rgb': np.concatenate(self.buffer['camera0_rgb'][-self.img_obs_horizon:], axis=0),
            'robot0_eef_pos': np.concatenate(self.buffer['robot0_eef_pos'][-self.low_dim_obs_horizon:], axis=0),
            'robot0_eef_rot_axis_angle': np.concatenate(self.buffer['robot0_eef_rot_axis_angle'][-self.low_dim_obs_horizon:], axis=0),
            'robot0_gripper_width': np.concatenate(self.buffer['robot0_gripper_width'][-self.low_dim_obs_horizon:], axis=0),
        }


def get_image(ts, camera_names, target_size=None):
    """
    Extract and process images from environment observation.
    For UMI policies, images are resized to target_size if provided.
    """
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        
        # Resize image if target_size is provided (for UMI)
        if target_size is not None:
            # Convert to numpy for cv2 processing
            img_np = curr_image.transpose(1, 2, 0)  # c h w -> h w c
            img_resized = cv2.resize(img_np, (target_size[1], target_size[0]))  # cv2 expects (width, height)
            curr_image = img_resized.transpose(2, 0, 1)  # h w c -> c h w
        
        curr_images.append(curr_image)
    
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def convert_action_buffer_to_trajectory(action_buffer):
    """
    Convert UMI action buffer (list of 8D sim actions) to trajectory format for MPC.
    
    Args:
        action_buffer: List of numpy arrays, each of shape (8,) representing
                      [x, y, z, qw, qx, qy, qz, gripper] in sim format
    
    Returns:
        trajectory: numpy array of shape (N, 7) representing
                   [x, y, z, qw, qx, qy, qz] for MPC (excluding gripper)
    """
    if len(action_buffer) == 0:
        return None
    
    trajectory = []
    for action in action_buffer:
        # Extract position and quaternion (exclude gripper)
        pos = action[:3]  # xyz
        quat = action[3:7]  # qw, qx, qy, qz
        trajectory.append(np.concatenate([pos, quat]))
    
    return np.array(trajectory)  # Shape: (N, 7)

def extract_trajectory_positions(action_buffer, current_ee_pos):
    """Extract 3D positions from interpolated action buffer for trajectory visualization
    
    Args:
        action_buffer: List of interpolated actions, each with [x, y, z, qw, qx, qy, qz, gripper]
        current_ee_pos: Current end effector tip position [x, y, z]
    
    Returns:
        Tuple of (trajectory_points, extrapolated_start_idx) where extrapolated_start_idx 
        indicates where extrapolated trajectory begins (None if no extrapolation)
    """
    trajectory_points = [current_ee_pos.copy()]  # Start with current tip position
    
    # Extract positions from each interpolated action in the buffer and project to tip
    for action in action_buffer:
        if len(action) >= 8:
            # Get target EE base position and orientation (already in sim format)
            target_ee_pos = action[:3]  # EE base position
            target_ee_quat = action[3:7]  # [qw, qx, qy, qz]
            
            # Project to tip position using the same transformation
            target_tip_pos = transform_to_tip(target_ee_pos, target_ee_quat, tip_offset=0.20)
            trajectory_points.append(target_tip_pos)
    
    return trajectory_points


def transform_to_tip(ee_pos, ee_quat, tip_offset=0.20):
    """Transform EE base position to tip position
    
    Args:
        ee_pos: EE base position [x, y, z]
        ee_quat: EE orientation quaternion [qw, qx, qy, qz]
        tip_offset: Distance to project forward along EE X-axis
    
    Returns:
        numpy array: Tip position [x, y, z]
    """
    from scipy.spatial.transform import Rotation as R
    
    # Convert quaternion to rotation matrix
    # Convert from [qw, qx, qy, qz] to scipy format [qx, qy, qz, qw]
    quat_scipy = np.array([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
    ee_rot_matrix = R.from_quat(quat_scipy).as_matrix()
    
    # Project forward along EE local X-axis to reach gripper tip
    # Based on XML: gripper fingers are ~0.08m, ee_tool is 0.14m, so use 0.16m
    tip_offset_local = np.array([tip_offset, 0.0, 0.0])
    tip_offset_world = ee_rot_matrix @ tip_offset_local
    tip_pos = ee_pos + tip_offset_world
    
    return tip_pos


def add_trajectory_to_viewer(viewer, trajectory_points, actual_trajectory_points=None, extrapolated_start_idx=None, candidate_trajectories=None, pregrad_trajectory=None, candidate_trajectories_vanilla=None, candidate_trajectories_guided=None):
    """Add trajectory visualization to MuJoCo viewer user scene
    
    Args:
        viewer: MuJoCo passive viewer handle
        trajectory_points: List of 3D positions for planned trajectory
        actual_trajectory_points: List of 3D positions for actual executed trajectory
        extrapolated_start_idx: Index where extrapolated trajectory begins (None if no extrapolation)
        candidate_trajectories: List of candidate trajectory position lists (non-selected trajectories)
        pregrad_trajectory: List of 3D positions for pregrad trajectory (before refinement)
    """
    if not viewer:
        return
        
    # Clear existing user geometries
    viewer.user_scn.ngeom = 0
    geom_idx = 0
    
    # Add vanilla figure trajectories (gradient blue/cyan scheme)
    if candidate_trajectories_vanilla:
        for candidate_points in candidate_trajectories_vanilla:
            if not candidate_points:
                continue
            n_pts = len(candidate_points)
            for i, pos in enumerate(candidate_points):
                if geom_idx >= len(viewer.user_scn.geoms):
                    break  # Don't exceed available geometry slots
                # Gradient across trajectory from light cyan to deep blue
                t = float(i) / float(max(n_pts - 1, 1))
                start_rgba = np.array([0.70, 0.90, 1.00, 0.2], dtype=np.float32)
                end_rgba   = np.array([0.05, 0.25, 1.00, 0.35], dtype=np.float32)
                rgba = start_rgba + t * (end_rgba - start_rgba)
                size = max(0.001, 0.002 - i * 0.0001)
                size_array = np.array([size, 0, 0], dtype=np.float64)
                pos_array = np.array(pos, dtype=np.float64).reshape(3, 1)
                mat_array = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
                rgba_reshaped = rgba.reshape(4, 1)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size_array,
                    pos=pos_array,
                    mat=mat_array,
                    rgba=rgba_reshaped
                )
                geom_idx += 1

    # Add guided figure trajectories (gradient pink/red scheme)
    if candidate_trajectories_guided:
        for candidate_points in candidate_trajectories_guided:
            if not candidate_points:
                continue
            n_pts = len(candidate_points)
            for i, pos in enumerate(candidate_points):
                if geom_idx >= len(viewer.user_scn.geoms):
                    break  # Don't exceed available geometry slots
                # Gradient across trajectory from light pink to deep magenta/red
                t = float(i) / float(max(n_pts - 1, 1))
                start_rgba = np.array([1.00, 0.82, 0.88, 0.2], dtype=np.float32)
                end_rgba   = np.array([0.95, 0.10, 0.28, 0.35], dtype=np.float32)
                rgba = start_rgba + t * (end_rgba - start_rgba)
                size = max(0.001, 0.002 - i * 0.0001)
                size_array = np.array([size, 0, 0], dtype=np.float64)
                pos_array = np.array(pos, dtype=np.float64).reshape(3, 1)
                mat_array = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
                rgba_reshaped = rgba.reshape(4, 1)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size_array,
                    pos=pos_array,
                    mat=mat_array,
                    rgba=rgba_reshaped
                )
                geom_idx += 1

    # Add generic candidate trajectories (legacy vis_traj) so they appear behind main trajectory
    if candidate_trajectories:
        for candidate_points in candidate_trajectories:
            if not candidate_points:
                continue
            for i, pos in enumerate(candidate_points):
                if geom_idx >= len(viewer.user_scn.geoms):
                    break  # Don't exceed available geometry slots
                
                # Semi-transparent version of main trajectory colors (30% opacity)
                if i == 0:
                    rgba = np.array([0, 1, 0, 0.25], dtype=np.float32)  # Semi-transparent green
                elif i <= 3:
                    rgba = np.array([1, 1, 0, 0.2], dtype=np.float32)   # Semi-transparent yellow
                else:
                    rgba = np.array([1, 0, 0, 0.15], dtype=np.float32)  # Semi-transparent red
                
                # Quarter size of main trajectory (0.008 -> 0.002)
                size = max(0.001, 0.002 - i * 0.0001)
                size_array = np.array([size, 0, 0], dtype=np.float64)
                
                # Ensure pos is proper numpy array
                pos_array = np.array(pos, dtype=np.float64).reshape(3, 1)
                mat_array = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
                rgba_reshaped = rgba.reshape(4, 1)
                
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[geom_idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size_array,
                    pos=pos_array,
                    mat=mat_array,
                    rgba=rgba_reshaped
                )
                geom_idx += 1
    
    # Add pregrad trajectory (distinct visualization for guidance mode)
    if pregrad_trajectory:
        for i, pos in enumerate(pregrad_trajectory):
            if geom_idx >= len(viewer.user_scn.geoms):
                break  # Don't exceed available geometry slots
            
            # Super light pink color scheme for pregrad trajectory
            if i == 0:
                rgba = np.array([1.0, 0.9, 0.9, 0.5], dtype=np.float32)  # Super light pink for start
            elif i <= 3:
                rgba = np.array([1.0, 0.8, 0.8, 0.4], dtype=np.float32)  # Light pink for near
            else:
                rgba = np.array([1.0, 0.7, 0.7, 0.3], dtype=np.float32)  # Super light pink for far
            
            base_size = 0.006
            size = max(0.002, base_size - i * 0.0005)
            size_array = np.array([size, 0, 0], dtype=np.float64)
            
            # Ensure pos is proper numpy array
            pos_array = np.array(pos, dtype=np.float64).reshape(3, 1)
            mat_array = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba_reshaped = rgba.reshape(4, 1)
            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size_array,
                pos=pos_array,
                mat=mat_array,
                rgba=rgba_reshaped
            )
            geom_idx += 1

    # Add planned trajectory spheres (colored)
    if trajectory_points:
        for i, pos in enumerate(trajectory_points):
            if geom_idx >= len(viewer.user_scn.geoms):
                break  # Don't exceed available geometry slots
            
            # Determine if this point is extrapolated
            is_extrapolated = extrapolated_start_idx is not None and i >= extrapolated_start_idx
            
            # Color coding: current position (green), near future (yellow), far future (red)
            # Extrapolated points use more transparent/different colors
            if i == 0:
                rgba = np.array([0, 1, 0, 0.8], dtype=np.float32)  # Green for current position
            elif is_extrapolated:
                # rgba = np.array([0, 0.5, 1, 0.3], dtype=np.float32)  # Cyan with low opacity for extrapolated
                pass
            elif i <= 3:
                rgba = np.array([1, 1, 0, 0.6], dtype=np.float32)  # Yellow for near future
            else:
                rgba = np.array([1, 0, 0, 0.4], dtype=np.float32)  # Red for far future
                
            base_size = 0.003 if is_extrapolated else 0.008
            size = max(0.001, base_size - i * 0.0005)
            size_array = np.array([size, 0, 0], dtype=np.float64)
            
            # Ensure pos is proper numpy array
            pos_array = np.array(pos, dtype=np.float64).reshape(3, 1)
            mat_array = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba_reshaped = rgba.reshape(4, 1)
            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size_array,
                pos=pos_array,
                mat=mat_array,
                rgba=rgba_reshaped
            )
            geom_idx += 1
    
    # Add actual trajectory dots (small grey persistent dots)
    if actual_trajectory_points:
        for i, pos in enumerate(actual_trajectory_points):
            if geom_idx >= len(viewer.user_scn.geoms):
                break  # Don't exceed available geometry slots
                
            # Small grey dots for actual trajectory
            rgba = np.array([0.5, 0.5, 0.5, 0.7], dtype=np.float32)  # Grey with some transparency
            size = 0.001  # Small consistent size
            size_array = np.array([size, 0, 0], dtype=np.float64)
            
            # Ensure pos is proper numpy array
            pos_array = np.array(pos, dtype=np.float64).reshape(3, 1)
            mat_array = np.eye(3, dtype=np.float64).flatten().reshape(9, 1)
            rgba_reshaped = rgba.reshape(4, 1)
            
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size_array,
                pos=pos_array,
                mat=mat_array,
                rgba=rgba_reshaped
            )
            geom_idx += 1
    
    viewer.user_scn.ngeom = geom_idx


def _rotation_matrix_from_z_axis(target_axis: np.ndarray) -> np.ndarray:
    """Build a rotation matrix whose third column (Z axis) aligns with target_axis."""
    axis = np.asarray(target_axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    z_axis = axis / norm
    # Choose an arbitrary vector not parallel to z_axis
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, z_axis)) > 0.95:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(tmp, z_axis)
    x_axis /= max(1e-12, np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    # Columns are the basis vectors
    Rm = np.stack([x_axis, y_axis, z_axis], axis=1)
    return Rm


def _add_vector_arrow(viewer, start_pos: np.ndarray, vec: np.ndarray, rgba: np.ndarray, radius: float) -> int:
    """Draw an outward-pointing arrow: body cylinder + short thicker capsule head.

    The arrow starts at start_pos and points along vec. Total drawn length is slightly
    reduced for aesthetics and to avoid clipping.
    """
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return 0

    # Slightly reduce overall length
    shrink = 0.85
    draw_vec = vec * shrink
    L = float(np.linalg.norm(draw_vec))
    unit = draw_vec / L
    rot = _rotation_matrix_from_z_axis(draw_vec)

    # Split into body and head
    head_len = float(np.clip(0.20 * L, 0.015, 0.05))
    body_len = max(1e-6, L - head_len)

    geoms_used = 0
    gidx = int(viewer.user_scn.ngeom)
    mat = np.array(rot, dtype=np.float64).flatten().reshape(9, 1)
    rgba_arr = np.array(rgba, dtype=np.float32).reshape(4, 1)

    # Body cylinder: from base to start + body_len
    body_center = start_pos + unit * (0.5 * body_len)
    size_body = np.array([radius, 0.5 * body_len, 0.0], dtype=np.float64)
    pos_body = np.array(body_center, dtype=np.float64).reshape(3, 1)
    if gidx < len(viewer.user_scn.geoms):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[gidx],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=size_body,
            pos=pos_body,
            mat=mat,
            rgba=rgba_arr,
        )
        gidx += 1
        geoms_used += 1

    # Tapered arrow head approximation using multiple short cylinders with decreasing radius
    tip_pos = start_pos + unit * L
    segments = 8
    for i in range(segments):
        # t from 0 at base of head to 1 at tip
        t0 = i / segments
        t1 = (i + 1) / segments
        # Segment center along the head portion
        seg_center_dist = (body_len + (t0 + t1) * 0.5 * head_len)
        seg_center = start_pos + unit * seg_center_dist
        # Segment half-length
        seg_half = 0.5 * (t1 - t0) * head_len
        # Radius tapers linearly toward tip
        seg_radius = radius * (2.5 - 2.0 * t0)  # from 2.5*r downwards
        size_seg = np.array([seg_radius, max(1e-6, seg_half), 0.0], dtype=np.float64)
        pos_seg = np.array(seg_center, dtype=np.float64).reshape(3, 1)
        if gidx < len(viewer.user_scn.geoms):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[gidx],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=size_seg,
                pos=pos_seg,
                mat=mat,
                rgba=rgba_arr,
            )
            gidx += 1
            geoms_used += 1

    viewer.user_scn.ngeom = gidx
    return geoms_used


def add_disturbance_arrows(viewer, physics, wind_vec: np.ndarray, torque_vec: np.ndarray,
                           wind_scale: float = 0.07, torque_scale: float = 0.10,
                           wind_color=(1.0, 0.0, 0.0, 0.9), torque_color=(1.0, 0.0, 0.0, 0.9),
                           wind_radius: float = 0.012, torque_radius: float = 0.009):
    """Append disturbance arrows (force and torque) to viewer user scene.

    Arrows originate from the drone base (`body name: base`).
    The user scene is not cleared here; this appends geoms after any existing ones.
    """
    if viewer is None or physics is None:
        return
    # Resolve base body id robustly across XML variants
    def _resolve_base_body_id(phys):
        # Try common names
        for name in ("base", "base_link", "arm_base_link"):
            try:
                return phys.model.body(name).id
            except KeyError:
                pass
        # Fallback: climb up from 'ee' to its highest non-world ancestor
        try:
            ee_bid = phys.model.body("ee").id
        except KeyError:
            # Last resort: use body 0 ('world')
            return 0
        parent_ids = phys.model.body_parentid
        pid = int(parent_ids[ee_bid])
        last_valid = ee_bid
        while pid > 0:
            last_valid = pid
            pid = int(parent_ids[pid])
        return last_valid

    base_bid = _resolve_base_body_id(physics)
    base_pos = physics.data.xpos[int(base_bid)].copy()

    # Append arrows after existing user geoms
    # Wind (external force felt by drone). Flip sign so it points outward from base as experienced.
    wind_vec = -np.asarray(wind_vec, dtype=np.float64) * float(wind_scale)
    _add_vector_arrow(viewer, base_pos, wind_vec, np.array(wind_color, dtype=np.float32), wind_radius)

    # Torque arrow (visualised as vector too); flip sign for consistency with "felt" direction
    torque_vec = -np.asarray(torque_vec, dtype=np.float64) * float(torque_scale)
    _add_vector_arrow(viewer, base_pos, torque_vec, np.array(torque_color, dtype=np.float32), torque_radius)

def get_ee_pos(physics):
    """Get current end effector tip position from physics state
    
    Args:
        physics: DM Control physics object
    
    Returns:
        numpy array: [x, y, z] position of end effector tip
    """
    # Get the 'ee' body (main EE reference frame from XML)
    ee_body_id = physics.model.body('ee').id
    ee_pos = physics.data.xpos[ee_body_id].copy()
    ee_rot_mat = physics.data.xmat[ee_body_id].reshape(3, 3).copy()
    
    # Project forward along EE local X-axis to reach gripper tip
    # Based on XML: gripper fingers are ~0.08m, ee_tool is 0.14m, so use 0.16m
    tip_offset = 0.20  # meters
    tip_offset_local = np.array([tip_offset, 0.0, 0.0])
    tip_offset_world = ee_rot_mat @ tip_offset_local
    ee_tip_pos = ee_pos + tip_offset_world
    
    return ee_tip_pos


def interpolate_action_sequence(actions, target_horizon):
    """
    Interpolate between action waypoints to create smooth trajectories.
    
    Args:
        actions: Array of shape (N, 7) containing [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, gripper]
        target_horizon: Total number of interpolated steps to generate (effective_action_horizon * obs_down_sample_steps)
        current_ee_pos: Current end effector position [x, y, z] to use as starting point
        current_ee_quat: Current end effector quaternion [qw, qx, qy, qz] (optional)
    
    Returns:
        List of interpolated 8D sim actions [x, y, z, qw, qx, qy, qz, gripper]
    """
    from scipy.spatial.transform import Rotation as R, Slerp
    
    if len(actions) == 0:
        return []
    
    # Convert actions to sim format first
    sim_actions = []
    for action in actions:
        xyz = action[:3]
        axis_angle = action[3:6]
        grip = action[6:7]
        
        # Convert axis-angle to quaternion
        if np.linalg.norm(axis_angle) > 1e-6:
            quat_xyzw = R.from_rotvec(axis_angle).as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        else:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        
        sim_action = np.concatenate([xyz, quat_wxyz, grip])  # [x, y, z, qw, qx, qy, qz, gripper]
        sim_actions.append(sim_action)

    waypoints = sim_actions#[2:]  # do NOT prepend current EE pose
    
    if len(waypoints) < 2:
        # If we have less than 2 waypoints, just repeat the single waypoint
        if len(waypoints) == 1:
            return [waypoints[0]] * target_horizon
        else:
            return []
    
    # Create evenly spaced interpolation across the entire trajectory
    interpolated_buffer = []
    
    # Create parameter values from 0 to 1 across the target horizon
    for i in range(target_horizon):
        t_global = i / (target_horizon - 1) if target_horizon > 1 else 0.0  # t ∈ [0, 1]
        
        # Find which segment this t_global falls into
        segment_length = 1.0 / (len(waypoints) - 1)
        segment_idx = min(int(t_global / segment_length), len(waypoints) - 2)
        
        # Local t within the current segment
        t_local = (t_global - segment_idx * segment_length) / segment_length
        t_local = np.clip(t_local, 0.0, 1.0)
        
        # Get the waypoints for this segment
        start_action = waypoints[segment_idx]
        end_action = waypoints[segment_idx + 1]
        
        # Extract components
        start_pos = start_action[:3]
        start_quat_wxyz = start_action[3:7]
        start_gripper = start_action[7:8]
        
        end_pos = end_action[:3]
        end_quat_wxyz = end_action[3:7]
        end_gripper = end_action[7:8]
        
        # Linear interpolation for position and gripper
        interp_pos = start_pos + t_local * (end_pos - start_pos)
        interp_gripper = start_gripper + t_local * (end_gripper - start_gripper)
        
        # SLERP for quaternion interpolation
        # Convert to scipy quaternion format [x, y, z, w]
        start_quat_xyzw = np.array([start_quat_wxyz[1], start_quat_wxyz[2], start_quat_wxyz[3], start_quat_wxyz[0]])
        end_quat_xyzw = np.array([end_quat_wxyz[1], end_quat_wxyz[2], end_quat_wxyz[3], end_quat_wxyz[0]])
        
        # Create rotation objects
        start_rot = R.from_quat(start_quat_xyzw)
        end_rot = R.from_quat(end_quat_xyzw)
        
        # SLERP interpolation
        slerp = Slerp([0, 1], R.concatenate([start_rot, end_rot]))
        interp_rot = slerp(t_local)
        interp_quat_xyzw = interp_rot.as_quat()
        
        # Convert back to [w, x, y, z] format
        interp_quat_wxyz = np.array([interp_quat_xyzw[3], interp_quat_xyzw[0], interp_quat_xyzw[1], interp_quat_xyzw[2]])
        
        # Combine into full action
        interp_action = np.concatenate([interp_pos, interp_quat_wxyz, interp_gripper])
        interpolated_buffer.append(interp_action)
    
    return interpolated_buffer


def extrapolate_action_buffer(action_buffer, target_size, num_steps_for_extrapolation=4):
    """
    Extrapolate action buffer when it's shorter than expected by continuing the motion trend.
    
    Args:
        action_buffer: Current action buffer (list of 8D sim actions)
        target_size: Expected buffer size (e.g., 32)
        num_steps_for_extrapolation: Number of recent steps to use for trend estimation
    
    Returns:
        Extended action buffer with extrapolated actions
    """
    from scipy.spatial.transform import Rotation as R
    
    if len(action_buffer) >= target_size or len(action_buffer) < 2:
        return action_buffer  # No extrapolation needed or not enough data
    
    # Get the last few actions to estimate trend
    recent_actions = action_buffer[-min(num_steps_for_extrapolation, len(action_buffer)):]
    
    if len(recent_actions) < 2:
        # Not enough data for trend, just repeat last action
        last_action = action_buffer[-1].copy()
        extrapolated_buffer = action_buffer.copy()
        for _ in range(target_size - len(action_buffer)):
            extrapolated_buffer.append(last_action.copy())
        return extrapolated_buffer
    
    # Calculate position and gripper velocity
    pos_velocities = []
    gripper_velocities = []
    
    for i in range(1, len(recent_actions)):
        prev_action = recent_actions[i-1]
        curr_action = recent_actions[i]
        
        pos_vel = curr_action[:3] - prev_action[:3]  # position velocity
        gripper_vel = curr_action[7:8] - prev_action[7:8]  # gripper velocity
        
        pos_velocities.append(pos_vel)
        gripper_velocities.append(gripper_vel)
    
    # Average the velocities for smoother extrapolation
    avg_pos_vel = np.mean(pos_velocities, axis=0)
    avg_gripper_vel = np.mean(gripper_velocities, axis=0)
    
    # For quaternion, we'll use the last quaternion (assuming stable orientation)
    last_quat = action_buffer[-1][3:7].copy()
    
    # Generate extrapolated actions
    extrapolated_buffer = action_buffer.copy()
    last_action = action_buffer[-1].copy()
    
    steps_to_add = target_size - len(action_buffer)
    for step in range(1, steps_to_add + 1):
        # Apply exponential decay to velocities over extrapolation steps
        decay_factor = 1.0 ** step  # Exponential decay (adjust factor as needed)
        
        # Extrapolate position with decay
        new_pos = last_action[:3] + avg_pos_vel * step * decay_factor
        
        # Keep orientation constant (could add small rotation if needed)
        new_quat = last_quat.copy()
        
        # Extrapolate gripper with decay (with clamping)
        new_gripper = np.clip(last_action[7:8] + avg_gripper_vel * step * decay_factor, 0.0, 1.0)
        
        # Create new action
        extrapolated_action = np.concatenate([new_pos, new_quat, new_gripper])
        extrapolated_buffer.append(extrapolated_action)
    
    return extrapolated_buffer


# Custom environment wrapper to pass trajectory to before_step
class TrajectoryAwareEnvWrapper:
    """Wrapper that allows passing trajectory information to the environment's before_step method."""
    
    def __init__(self, env):
        self.env = env
        self._pending_trajectory = None
    
    def set_trajectory(self, trajectory):
        """Set trajectory to be passed to next before_step call."""
        self._pending_trajectory = trajectory
    
    def step(self, action):
        """Step environment, passing trajectory if available."""
        # Override the task's before_step method temporarily
        original_before_step = self.env.task.before_step
        
        def before_step_with_trajectory(action, physics, trajectory=None):
            # Use pending trajectory if available, otherwise use passed trajectory
            traj_to_use = self._pending_trajectory if self._pending_trajectory is not None else trajectory
            return original_before_step(action, physics, trajectory=traj_to_use)
        
        # Temporarily replace the method
        self.env.task.before_step = before_step_with_trajectory
        
        try:
            # Call the original step
            timestep = self.env.step(action)
            # Clear the pending trajectory after use
            self._pending_trajectory = None
            return timestep
        finally:
            # Restore original method
            self.env.task.before_step = original_before_step
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped environment."""
        return getattr(self.env, name)
