#!/usr/bin/env python3
"""
Convert HDF5 dataset to UMI zarr format.

Usage:
python convert_hdf5_to_umi_zarr.py \
    --input_dir data/bc/sim_peg_in_hole_human/demonstration/04110424 \
    --output_path umi_dataset.zarr.zip \
    --camera_name ee \
    --image_size 224
"""

import os
import h5py
import zarr
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import glob
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()

def quaternion_to_axis_angle(quat):
    """Convert quaternion (w,x,y,z) to axis-angle representation."""
    # Ensure quaternion is normalized
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    
    # Convert to rotation object using scipy (expects x,y,z,w format)
    rotation = R.from_quat(quat[..., [1, 2, 3, 0]])  # Convert from w,x,y,z to x,y,z,w
    
    # Convert to axis-angle (rotvec)
    axis_angle = rotation.as_rotvec()
    return axis_angle

def resize_image(image, target_size=224):
    """Resize image to target size keeping HWC format."""
    # image is (H, W, C), keep it as HWC for UMI format
    resized = cv2.resize(image, (target_size, target_size))
    return resized

def load_episode_data(hdf5_path, camera_name='ee'):
    """Load data from a single episode HDF5 file."""
    with h5py.File(hdf5_path, 'r') as f:
        # Load observations
        qpos = f['/observations/qpos'][()]  # (T, 8): [ee_pos(3), ee_quat(4), gripper(1)]
        images = f[f'/observations/images/{camera_name}'][()]  # (T, H, W, C)
        
        # Load actions
        action = f['/action'][()]  # (T, 8): similar structure to qpos
        
        # Extract components from qpos (observations)
        ee_pos = qpos[:, :3]  # (T, 3)
        ee_quat = qpos[:, 3:7]  # (T, 4) - quaternion in w,x,y,z format
        gripper = qpos[:, 7:8]  # (T, 1)
        
        # Extract components from action
        action_ee_pos = action[:, :3]  # (T, 3)
        action_ee_quat = action[:, 3:7]  # (T, 4) - quaternion in w,x,y,z format
        action_gripper = action[:, 7:8]  # (T, 1)
        
        # Convert quaternions to axis-angle
        ee_rot_axis_angle = quaternion_to_axis_angle(ee_quat)  # (T, 3)
        action_ee_rot_axis_angle = quaternion_to_axis_angle(action_ee_quat)  # (T, 3)
        
        return {
            'ee_pos': ee_pos.astype(np.float32),
            'ee_rot_axis_angle': ee_rot_axis_angle.astype(np.float32),
            'gripper': gripper.astype(np.float32),
            'action_ee_pos': action_ee_pos.astype(np.float32),
            'action_ee_rot_axis_angle': action_ee_rot_axis_angle.astype(np.float32),
            'action_gripper': action_gripper.astype(np.float32),
            'images': images
        }

def convert_dataset(input_dir, output_path, camera_name='ee', image_size=224):
    """Convert HDF5 dataset to UMI zarr format."""
    input_path = Path(input_dir)
    
    # Find all episode files
    episode_files = sorted(glob.glob(str(input_path / "episode_*.hdf5")))
    print(f"Found {len(episode_files)} episodes")
    
    if len(episode_files) == 0:
        raise ValueError(f"No episode files found in {input_dir}")
    
    # Create replay buffer with memory store
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())
    
    print("Loading episodes...")
    for episode_file in tqdm(episode_files):
        episode_data = load_episode_data(episode_file, camera_name)
        episode_length = len(episode_data['ee_pos'])
        
        # Process images
        episode_images = []
        for img in episode_data['images']:
            resized_img = resize_image(img, image_size)
            episode_images.append(resized_img)
        episode_images = np.stack(episode_images)
        
        # Ensure images are uint8 (0-255 range)
        if episode_images.dtype != np.uint8:
            episode_images = (episode_images * 255).astype(np.uint8)
        
        # Process actions - concatenate [pos(3), axis_angle(3), gripper(1)] = (7,)
        episode_actions = np.concatenate([
            episode_data['action_ee_pos'],
            episode_data['action_ee_rot_axis_angle'],
            episode_data['action_gripper']
        ], axis=1)
        
        # Create demo start and end poses (6D: pos + axis_angle)
        demo_start_pose = np.concatenate([
            episode_data['ee_pos'][0:1],  # First timestep position (1, 3)
            episode_data['ee_rot_axis_angle'][0:1]  # First timestep rotation (1, 3)
        ], axis=1)  # Shape: (1, 6)
        
        demo_end_pose = np.concatenate([
            episode_data['ee_pos'][-1:],  # Last timestep position (1, 3)  
            episode_data['ee_rot_axis_angle'][-1:]  # Last timestep rotation (1, 3)
        ], axis=1)  # Shape: (1, 6)
        
        # Expand to match episode length for storage
        demo_start_poses = np.tile(demo_start_pose, (episode_length, 1))  # (T, 6)
        demo_end_poses = np.tile(demo_end_pose, (episode_length, 1))  # (T, 6)
        
        # Prepare episode data for adding to replay buffer
        episode_dict = {
            'robot0_eef_pos': episode_data['ee_pos'],
            'robot0_eef_rot_axis_angle': episode_data['ee_rot_axis_angle'],
            'robot0_gripper_width': episode_data['gripper'],
            'robot0_demo_start_pose': demo_start_poses,
            'robot0_demo_end_pose': demo_end_poses,
            'camera0_rgb': episode_images,
            'action': episode_actions
        }
        
        # Define compression for each data type
        compressors = {
            'robot0_eef_pos': None,
            'robot0_eef_rot_axis_angle': None,
            'robot0_gripper_width': None,
            'robot0_demo_start_pose': None,
            'robot0_demo_end_pose': None,
            'action': None,
            'camera0_rgb': Jpeg2k(level=50)
        }
        
        # Define chunks for each data type
        chunks = {
            'robot0_eef_pos': episode_data['ee_pos'].shape,
            'robot0_eef_rot_axis_angle': episode_data['ee_rot_axis_angle'].shape,
            'robot0_gripper_width': episode_data['gripper'].shape,
            'robot0_demo_start_pose': demo_start_poses.shape,
            'robot0_demo_end_pose': demo_end_poses.shape,
            'action': episode_actions.shape,
            'camera0_rgb': (1,) + episode_images.shape[1:]  # Chunk per timestep for images
        }
        
        # Add episode to replay buffer
        replay_buffer.add_episode(
            data=episode_dict,
            chunks=chunks,
            compressors=compressors
        )
    
    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"Total episodes: {replay_buffer.n_episodes}")
    print(f"Data shapes:")
    for key in replay_buffer.keys():
        print(f"  {key}: {replay_buffer[key].shape}")
    
    # Verify action dimensionality
    action_shape = replay_buffer['action'].shape
    print(f"Action shape: {action_shape}")
    if action_shape[-1] != 7:
        print(f"WARNING: Expected action dimension 7, got {action_shape[-1]}")
    
    # Save to disk using ReplayBuffer's built-in method
    print(f"Saving dataset to: {output_path}")
    if output_path.endswith('.zip'):
        # Use ZipStore directly for .zip files
        with zarr.ZipStore(output_path, mode='w') as zip_store:
            replay_buffer.save_to_store(
                store=zip_store
            )
        print(f"Dataset converted and saved to: {output_path}")
    else:
        replay_buffer.save_to_path(output_path)
        print(f"Dataset converted and saved to: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 dataset to UMI zarr format')
    parser.add_argument('--input_dir', required=True, 
                      help='Input directory containing episode_*.hdf5 files')
    parser.add_argument('--output_path', required=True,
                      help='Output zarr file path (e.g., dataset.zarr.zip)')
    parser.add_argument('--camera_name', default='ee',
                      help='Camera name in HDF5 files (default: ee)')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Target image size (default: 224)')
    
    args = parser.parse_args()
    
    convert_dataset(
        input_dir=args.input_dir,
        output_path=args.output_path,
        camera_name=args.camera_name,
        image_size=args.image_size
    )

if __name__ == '__main__':
    main() 