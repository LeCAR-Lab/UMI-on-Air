#!/usr/bin/env python3
"""
Verify the converted UMI zarr dataset structure and contents.
"""

import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

# Register codecs first
register_codecs()

def verify_dataset(zarr_path):
    """Verify the dataset structure and contents."""
    print(f"Verifying dataset: {zarr_path}")
    print("=" * 50)
    
    # Load the dataset directly with zarr first to inspect structure
    print("Loading zarr structure...")
    with zarr.ZipStore(zarr_path, mode='r') as zip_store:
        root = zarr.group(store=zip_store)
        print(f"Root group tree:")
        print(root.tree())
        print()
        
        # Check if data and meta groups exist
        if 'data' in root:
            print("Data group contents:")
            data_group = root['data']
            for key in data_group.keys():
                arr = data_group[key]
                print(f"  {key}:")
                print(f"    Shape: {arr.shape}")
                print(f"    Dtype: {arr.dtype}")
                print(f"    Chunks: {arr.chunks}")
                print(f"    Compressor: {arr.compressor}")
                
                # Try to read a small sample to verify accessibility
                try:
                    if 'rgb' in key:
                        sample = arr[0]  # Just first timestep
                        print(f"    Sample values: min={sample.min()}, max={sample.max()}")
                    else:
                        sample = arr[:5]  # First 5 timesteps
                        print(f"    Sample values: {sample}")
                except Exception as e:
                    print(f"    ⚠️  Error reading data: {e}")
                print()
        
        if 'meta' in root:
            print("Meta group contents:")
            meta_group = root['meta']
            for key in meta_group.keys():
                arr = meta_group[key]
                print(f"  {key}: {arr[:]}")
            print()
    
    # Now try to load with ReplayBuffer
    try:
        print("Loading with ReplayBuffer...")
        with zarr.ZipStore(zarr_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, 
                store=zarr.MemoryStore()
            )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Backend: {replay_buffer.backend}")
        print(f"Total steps: {replay_buffer.n_steps}")
        print(f"Total episodes: {replay_buffer.n_episodes}")
        print()
        
        # Check episode structure
        episode_ends = replay_buffer.episode_ends[:]
        episode_lengths = replay_buffer.episode_lengths[:]
        print("Episode structure:")
        print(f"  Episode ends: {episode_ends[:5]}..." if len(episode_ends) > 5 else f"  Episode ends: {episode_ends}")
        print(f"  Episode lengths - min: {episode_lengths.min()}, max: {episode_lengths.max()}, mean: {episode_lengths.mean():.1f}")
        print()
        
        # Verify expected UMI format
        expected_keys = {
            'robot0_eef_pos': (3,),
            'robot0_eef_rot_axis_angle': (3,),
            'robot0_gripper_width': (1,),
            'robot0_demo_start_pose': (6,),
            'robot0_demo_end_pose': (6,),
            'action': (7,)
        }
        
        # Optional keys (not all datasets have these)
        optional_keys = {
            'camera0_rgb': (224, 224, 3),
            'hole_location': (3,)
        }
        
        print("UMI format verification:")
        all_correct = True
        
        # Check required keys
        for key, expected_shape in expected_keys.items():
            if key not in replay_buffer.keys():
                print(f"  ❌ Missing required key: {key}")
                all_correct = False
            else:
                data_shape = replay_buffer[key].shape[1:]  # Skip time dimension
                if data_shape == expected_shape:
                    print(f"  ✅ {key}: {data_shape}")
                else:
                    print(f"  ❌ {key}: expected {expected_shape}, got {data_shape}")
                    all_correct = False
        
        # Check optional keys (just report, don't fail)
        for key, expected_shape in optional_keys.items():
            if key in replay_buffer.keys():
                data_shape = replay_buffer[key].shape[1:]  # Skip time dimension
                if data_shape == expected_shape:
                    print(f"  ✅ {key}: {data_shape} (optional)")
                else:
                    print(f"  ⚠️  {key}: expected {expected_shape}, got {data_shape} (optional)")
            else:
                print(f"  ➖ {key}: not present (optional)")
        
        if all_correct:
            print("\n🎉 Dataset verification passed! Ready for UMI training.")
            print(f"\nTo train with this dataset, use:")
            print(f"python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=your_dataset.zarr.zip")
        else:
            print("\n⚠️  Dataset has some format issues with required fields.")
        
        return all_correct
    
    except Exception as e:
        print(f"❌ Error loading with ReplayBuffer: {e}")
        print("The zarr structure looks correct but there might be codec issues.")
        return False

if __name__ == '__main__':
    # verify_dataset('/home/harsh/flyingumi/data_keyboard/bc/peg_in_hole_dataset.zarr.zip')
    verify_dataset('/home/harsh/flyingumi/data_mocap/mocap_dataset.zarr.zip') 