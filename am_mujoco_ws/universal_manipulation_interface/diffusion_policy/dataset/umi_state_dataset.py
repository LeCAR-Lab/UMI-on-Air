import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats, concatenate_normalizer, get_identity_normalizer_from_stat,
    get_image_identity_normalizer, get_range_normalizer_from_stat)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import pose_to_mat, mat_to_pose10d

register_codecs()

class UmiStateDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None
    ):
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        if cache_dir is None:
            # load into memory store
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, 
                    store=zarr.MemoryStore()
                )
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        self.num_robot = 0
        rgb_keys = list()
        lowdim_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            # solve obs type
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = list()
        for key in lowdim_keys:
            if not 'wrt' in key:
                self.sampler_lowdim_keys.append(key)
    
        for key in replay_buffer.keys():
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split('_')[0] + '_eef_pos'
                key_horizon[key] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[key] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[key] = shape_meta['obs'][query_key]['down_sample_steps']

        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # enumerate the dataset and save low_dim data
        data_cache = {key: list() for key in self.lowdim_keys + ['action']}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc='iterating dataset to get normalization'):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch['obs'][key]))
            data_cache['action'].append(copy.deepcopy(batch['action']))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            B, T, D = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(B*T, D)

        # action
        assert data_cache['action'].shape[-1] % self.num_robot == 0
        dim_a = data_cache['action'].shape[-1] // self.num_robot
        action_normalizers = list()
        for i in range(self.num_robot):
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a: i * dim_a + 3])))              # pos
            action_normalizers.append(get_identity_normalizer_from_stat(array_to_stats(data_cache['action'][..., i * dim_a + 3: (i + 1) * dim_a - 1]))) # rot
            action_normalizers.append(get_range_normalizer_from_stat(array_to_stats(data_cache['action'][..., (i + 1) * dim_a - 1: (i + 1) * dim_a])))  # gripper

        normalizer['action'] = concatenate_normalizer(action_normalizers)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])

            if key.endswith('pos') or 'pos_wrt' in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('pos_abs'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('rot_axis_angle') or 'rot_axis_angle_wrt' in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('gripper_width'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = dict()
        for key in self.rgb_keys:
            if not key in data:
                continue
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]
        
        # generate relative pose between two ees
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if not f'robot{robot_id}_eef_pos_wrt{other_robot_id}' in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(np.concatenate([
                    obs_dict[f'robot{other_robot_id}_eef_pos'],
                    obs_dict[f'robot{other_robot_id}_eef_rot_axis_angle']
                ], axis=-1))
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep='relative',
                    backward=False)
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]
                
        # generate relative pose with respect to episode start
        for robot_id in range(self.num_robot):
            # HACK: add noise to episode start pose
            if (f'robot{other_robot_id}_eef_pos_wrt_start' not in self.shape_meta['obs']) and \
                (f'robot{other_robot_id}_eef_rot_axis_angle_wrt_start' not in self.shape_meta['obs']):
                continue
            
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = obs_dict[f'robot{robot_id}_demo_start_pose'][0]
            # HACK: add noise to episode start pose
            start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # HACK: add noise to episode start pose
            # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

        del_keys = list()
        for key in obs_dict:
            if key.endswith('_demo_start_pose') or key.endswith('_demo_end_pose'):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = list()
        for robot_id in range(self.num_robot):
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                obs_dict[f'robot{robot_id}_eef_pos'],
                obs_dict[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            action_mat = pose_to_mat(data['action'][...,7 * robot_id: 7 * robot_id + 6])
            
            # solve relative obs
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
            action_pose_mat = convert_pose_mat_rep(
                action_mat, 
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False)
        
            # convert pose to pos + rot6d representation
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
        
            action_gripper = data['action'][..., 7 * robot_id + 6: 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))

            # generate data
            obs_dict[f'robot{robot_id}_eef_pos'] = obs_pose[:,:3]
            obs_dict[f'robot{robot_id}_eef_rot_axis_angle'] = obs_pose[:,3:]
            
        data['action'] = np.concatenate(actions, axis=-1)
        
        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data

if __name__ == "__main__":
    import os
    
    # Set dataset path to the mocap dataset
    dataset_path = "/home/harsh/flyingumi/data_mocap/mocap_dataset.zarr.zip"
    
    # Create a simplified configuration for testing
    shape_meta = {
        "obs": {
            "cur_pos": {
                "shape": [3],
                "horizon": 2,
                "latency_steps": 6.25,
                "down_sample_steps": 4,
                "type": "low_dim",
                "ignore_by_policy": False
            },
            "hole_pos": {
                "shape": [3],
                "horizon": 2,
                "latency_steps": 6.25,
                "down_sample_steps": 4,
                "type": "low_dim",
                "ignore_by_policy": False
            },
            "robot0_eef_pos": {
                "shape": [3],
                "horizon": 2,
                "latency_steps": 6.25,
                "down_sample_steps": 4,
                "type": "low_dim",
                "ignore_by_policy": False
            },
            "robot0_eef_rot_axis_angle": {
                "raw_shape": [3],
                "shape": [6],
                "horizon": 2,
                "latency_steps": 6.25,
                "down_sample_steps": 4,
                "type": "low_dim",
                "rotation_rep": "rotation_6d",
                "ignore_by_policy": False
            },
            "robot0_gripper_width": {
                "shape": [1],
                "horizon": 2,
                "latency_steps": 5.25,
                "down_sample_steps": 4,
                "type": "low_dim",
                "ignore_by_policy": False
            },
            "robot0_eef_rot_axis_angle_wrt_start": {
                "raw_shape": [3],
                "shape": [6],
                "horizon": 2,
                "latency_steps": 6.25,
                "down_sample_steps": 4,
                "type": "low_dim",
                "ignore_by_policy": False
            }
        },
        "action": {
            "shape": [10],
            "horizon": 8,
            "latency_steps": 0,
            "down_sample_steps": 4,
            "rotation_rep": "rotation_6d"
        }
    }
    
    pose_repr = {
        "obs_pose_repr": "relative",
        "action_pose_repr": "relative"
    }
    
    # Create dataset
    try:
        print("Initializing dataset...")
        print(f"Loading dataset from: {dataset_path}")
        
        dataset = UmiStateDataset(
            shape_meta=shape_meta,
            dataset_path=dataset_path,
            pose_repr=pose_repr,
            action_padding=False,
            temporally_independent_normalization=False,
            repeat_frame_prob=0.0,
            seed=42,
            val_ratio=0.05
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Sample a few items from the dataset
        print("Sampling items from dataset...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print("Observation keys:")
            for key in sample['obs']:
                shape = sample['obs'][key].shape
                print(f"  {key}: {shape}, dtype: {sample['obs'][key].dtype}")
            
            print("Action shape:", sample['action'].shape, "dtype:", sample['action'].dtype)
        
        # Test getting normalizer
        print("\nGetting normalizer...")
        normalizer = dataset.get_normalizer()
        print("Normalizer type:", type(normalizer))
        
        # Check what's available in the normalizer
        print("\nNormalizer parameters:")
        for name, param in normalizer.named_parameters():
            print(f"  {name}: {param.shape}")
            
        print("\nNormalizer state dict keys:")
        for key in normalizer.state_dict().keys():
            print(f"  {key}")
        
        # Test normalizing data
        print("\nTesting normalizer on a sample...")
        sample = dataset[0]
        
        # Prepare data for normalization
        data_to_normalize = {
            **sample['obs'],
            'action': sample['action']
        }
        
        # Normalize the data
        normalized_data = normalizer.normalize(data_to_normalize)
        
        # Extract normalized observations and action
        normalized_obs = {k: normalized_data[k] for k in sample['obs']}
        normalized_action = normalized_data['action']
        
        print("Normalized observation keys:")
        for key in normalized_obs:
            print(f"  {key}: {normalized_obs[key].shape}")
        print("Normalized action shape:", normalized_action.shape)
        
        # Test unnormalization
        unnormalized_data = normalizer.unnormalize(normalized_data)
        
        # Check if unnormalization recovers the original data
        print("\nChecking unnormalization accuracy:")
        for key in sample['obs']:
            orig = sample['obs'][key]
            recovered = unnormalized_data[key]
            error = torch.abs(orig - recovered).mean().item()
            print(f"  {key} mean abs error: {error:.6f}")
            
        action_error = torch.abs(sample['action'] - unnormalized_data['action']).mean().item()
        print(f"  action mean abs error: {action_error:.6f}")
        
        # Test validation dataset
        val_dataset = dataset.get_validation_dataset()
        print(f"\nValidation dataset size: {len(val_dataset)}")
        
        # Check data distribution
        print("\nCalculating basic statistics on data...")
        import torch
        # Collect a batch of data
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=100,
            shuffle=True,
            num_workers=4
        )
        batch = next(iter(dataloader))
        
        # Print statistics for a few observation keys
        for key in ['robot0_eef_pos', 'robot0_gripper_width']:
            data = batch['obs'][key]
            # Flatten all dimensions except the last one
            flat_data = data.reshape(-1, data.shape[-1])
            print(f"\n{key} statistics:")
            print(f"  Mean: {flat_data.mean(dim=0)}")
            print(f"  Std: {flat_data.std(dim=0)}")
            print(f"  Min: {flat_data.min(dim=0).values}")
            print(f"  Max: {flat_data.max(dim=0).values}")
        
        # Print hole_pos statistics specifically
        if 'hole_pos' in batch['obs']:
            hole_data = batch['obs']['hole_pos']
            flat_hole = hole_data.reshape(-1, hole_data.shape[-1])
            print("\nhole_pos statistics:")
            print(f"  Shape: {hole_data.shape}")
            print(f"  Mean: {flat_hole.mean(dim=0)}")
            print(f"  Std: {flat_hole.std(dim=0)}")
            print(f"  Min: {flat_hole.min(dim=0).values}")
            print(f"  Max: {flat_hole.max(dim=0).values}")
        else:
            print("\nWARNING: hole_pos not found in observation data!")
            print("Available keys:", list(batch['obs'].keys()))
            
        # Print cur_pos statistics specifically
        if 'cur_pos' in batch['obs']:
            cur_data = batch['obs']['cur_pos']
            flat_cur = cur_data.reshape(-1, cur_data.shape[-1])
            print("\ncur_pos statistics:")
            print(f"  Shape: {cur_data.shape}")
            print(f"  Mean: {flat_cur.mean(dim=0)}")
            print(f"  Std: {flat_cur.std(dim=0)}")
            print(f"  Min: {flat_cur.min(dim=0).values}")
            print(f"  Max: {flat_cur.max(dim=0).values}")
            
            # Normalize and unnormalize to check values
            print("\nChecking cur_pos normalization/unnormalization:")
            # Get a sample of cur_pos data
            sample_cur_pos = {'cur_pos': batch['obs']['cur_pos'][0:1]}
            normalized_cur_pos = normalizer.normalize(sample_cur_pos)
            unnormalized_cur_pos = normalizer.unnormalize(normalized_cur_pos)
            
            print("Original values:")
            print(sample_cur_pos['cur_pos'])
            print("Normalized values:")
            print(normalized_cur_pos['cur_pos'])
            print("Unnormalized values:")
            print(unnormalized_cur_pos['cur_pos'])
            print("Difference (original - unnormalized):")
            print(torch.abs(sample_cur_pos['cur_pos'] - unnormalized_cur_pos['cur_pos']).mean())
        else:
            print("\nWARNING: cur_pos not found in observation data!")
            print("Available keys:", list(batch['obs'].keys()))
        
        # Print action statistics
        action_data = batch['action']
        flat_action = action_data.reshape(-1, action_data.shape[-1])
        print("\nAction statistics:")
        print(f"  Mean: {flat_action.mean(dim=0)}")
        print(f"  Std: {flat_action.std(dim=0)}")
        print(f"  Min: {flat_action.min(dim=0).values}")
        print(f"  Max: {flat_action.max(dim=0).values}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during dataset testing: {e}")
        import traceback
        traceback.print_exc()
