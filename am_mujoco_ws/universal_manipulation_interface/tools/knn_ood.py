import os
import sys
import argparse
import pathlib
import dill
import json
from typing import List, Tuple, Optional

import numpy as np
import torch

# Ensure imports resolve when running this script directly
UI_DIR = pathlib.Path(__file__).parent.parent  # .../universal_manipulation_interface
REPO_DIR = UI_DIR.parent                       # .../am_mujoco_ws
for p in [str(UI_DIR), str(REPO_DIR)]:
    if p not in sys.path:
        sys.path.append(p)
os.chdir(str(UI_DIR))

import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.umi_dataset import UmiDataset
from diffusion_policy.common.sampler import SequenceSampler
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


def resolve_ckpt_path(ckpt_path: str = None, ckpt_dir: str = None) -> str:
    if ckpt_path is not None:
        return ckpt_path
    if ckpt_dir is not None:
        latest = pathlib.Path(ckpt_dir).joinpath('checkpoints', 'latest.ckpt')
        if latest.exists():
            return str(latest)
        ckpt_dir_path = pathlib.Path(ckpt_dir).joinpath('checkpoints')
        if ckpt_dir_path.exists():
            all_ckpts = sorted(ckpt_dir_path.glob('*.ckpt'))
            if len(all_ckpts) > 0:
                return str(all_ckpts[-1])
    raise FileNotFoundError('Could not resolve checkpoint path. Provide --load_ckpt_file_path or --ckpt_dir.')


def load_policy_from_ckpt(ckpt_path: str):
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.get('use_ema', False):
        if hasattr(workspace, 'ema_model') and workspace.ema_model is not None:
            policy = workspace.ema_model

    policy.eval()
    if torch.cuda.is_available():
        policy.cuda()
    return policy, cfg


def build_dataset_from_cfg(cfg, val_ratio: float = 0.0, dataset_path_override: Optional[str] = None):
    shape_meta = cfg.task.shape_meta
    dataset_path = dataset_path_override if dataset_path_override is not None else cfg.task.dataset_path
    pose_repr = cfg.task.pose_repr

    dataset = UmiDataset(
        shape_meta=shape_meta,
        dataset_path=dataset_path,
        cache_dir=None,
        pose_repr=pose_repr,
        action_padding=False,
        temporally_independent_normalization=False,
        repeat_frame_prob=0.0,
        seed=42,
        val_ratio=val_ratio,
        max_duration=None,
    )
    return dataset


def restrict_dataset_to_first_eps(dataset: UmiDataset, eps: int) -> None:
    total_eps = dataset.replay_buffer.n_episodes
    use_eps = min(max(0, eps), total_eps)
    if use_eps == 0:
        return
    episode_mask = np.zeros(total_eps, dtype=bool)
    episode_mask[:use_eps] = True
    dataset.sampler = SequenceSampler(
        shape_meta=dataset.shape_meta,
        replay_buffer=dataset.replay_buffer,
        rgb_keys=dataset.rgb_keys,
        lowdim_keys=dataset.sampler_lowdim_keys,
        key_horizon=dataset.key_horizon,
        key_latency_steps=dataset.key_latency_steps,
        key_down_sample_steps=dataset.key_down_sample_steps,
        episode_mask=episode_mask,
        action_padding=dataset.action_padding,
        repeat_frame_prob=dataset.repeat_frame_prob,
        max_duration=dataset.max_duration,
    )


def encode_base_features(
    policy,
    cfg,
    dataset_path: str,
    out_dir: pathlib.Path,
    batch_size: int,
    num_workers: int,
    disable_aug: bool,
    max_samples: int,
    eps: int,
) -> np.ndarray:
    dataset = build_dataset_from_cfg(cfg, val_ratio=0.0, dataset_path_override=dataset_path)
    if eps and eps > 0:
        restrict_dataset_to_first_eps(dataset, eps)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    if disable_aug and hasattr(policy, 'obs_encoder'):
        import torch.nn as nn
        if hasattr(policy.obs_encoder, 'key_transform_map'):
            for k in list(policy.obs_encoder.key_transform_map.keys()):
                policy.obs_encoder.key_transform_map[k] = nn.Identity()

    # Determine feature dim
    with torch.no_grad():
        first_batch = next(iter(loader))
        obs = first_batch['obs']
        device = policy.device
        obs = dict_apply(obs, lambda x: x.to(device=device, non_blocking=True))
        nobs = policy.normalizer.normalize(obs)
        feats = policy.obs_encoder(nobs)
        feat_dim = int(np.prod(list(feats.shape[1:])))

    N = len(dataset)
    features_path = out_dir.joinpath('features_memmap.float32')
    features = np.memmap(features_path, mode='w+', dtype=np.float32, shape=(N, feat_dim))
    idx_base = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Encoding base features', unit='batch'):
            obs = batch['obs']
            B = next(iter(obs.values())).shape[0]
            if max_samples > 0 and (idx_base >= max_samples):
                break
            obs = dict_apply(obs, lambda x: x.to(device=policy.device, non_blocking=True))
            nobs = policy.normalizer.normalize(obs)
            enc = policy.obs_encoder(nobs)
            enc_np = enc.detach().cpu().numpy().reshape(B, -1).astype(np.float32)
            write_end = idx_base + B
            if max_samples > 0:
                write_end = min(write_end, max_samples)
                enc_np = enc_np[: write_end - idx_base]
            features[idx_base:write_end] = enc_np
            idx_base = write_end
            if max_samples > 0 and idx_base >= max_samples:
                break
    total = idx_base
    features.flush()
    base_features = np.array(features[:total])
    np.save(out_dir.joinpath('features.npy'), base_features)
    return base_features


def encode_special_dirs_features(
    policy,
    cfg,
    special_dirs: List[str],
    batch_size: int,
    max_samples: int,
) -> Tuple[np.ndarray, List[Tuple[str, int]], np.ndarray]:
    spec_feats_per_dir: List[np.ndarray] = []
    dir_slices: List[Tuple[str, int]] = []
    all_progress: List[np.ndarray] = []
    for dir_path in special_dirs:
        dpath = pathlib.Path(dir_path)
        npz_files = sorted(dpath.glob('episode_*.npz'))
        if len(npz_files) == 0:
            raise FileNotFoundError(f'No episode_*.npz found in {dpath}')
        spec_feats_list: List[np.ndarray] = []
        spec_prog_list: List[np.ndarray] = []
        total_written = 0
        for npz_path in tqdm(npz_files, desc=f'Encoding specials from {dpath.name}', unit='file'):
            data = np.load(npz_path)
            obs_np = {
                'camera0_rgb': data['camera0_rgb'],
                'robot0_eef_pos': data['robot0_eef_pos'],
                'robot0_eef_rot_axis_angle': data['robot0_eef_rot_axis_angle'],
                'robot0_gripper_width': data['robot0_gripper_width'],
            }
            prog_spec = data['progress'].astype(np.float32)
            Btot = obs_np['camera0_rgb'].shape[0]
            bs = max(1, min(batch_size, Btot))
            shape_meta = cfg.task.shape_meta['obs']
            expected_lowdim = list(getattr(policy.obs_encoder, 'low_dim_keys', []))
            frames_written_this_file = 0
            with torch.no_grad():
                for start in range(0, Btot, bs):
                    end = min(start + bs, Btot)
                    if max_samples > 0 and total_written >= max_samples:
                        break
                    obs_chunk = {
                        'camera0_rgb': torch.from_numpy(obs_np['camera0_rgb'][start:end]).float().to(policy.device)
                    }
                    for key in expected_lowdim:
                        if key in obs_np:
                            obs_chunk[key] = torch.from_numpy(obs_np[key][start:end]).float().to(policy.device)
                        else:
                            T = int(shape_meta[key]['horizon'])
                            D = int(shape_meta[key]['shape'][-1])
                            obs_chunk[key] = torch.zeros((end - start, T, D), dtype=torch.float32, device=policy.device)
                    nobs = policy.normalizer.normalize(obs_chunk)
                    enc = policy.obs_encoder(nobs)
                    enc_np = enc.detach().cpu().numpy().reshape(end - start, -1)
                    # Respect global max_samples by slicing within this file
                    if max_samples > 0 and (total_written + enc_np.shape[0] > max_samples):
                        keep = max(0, max_samples - total_written)
                        if keep <= 0:
                            break
                        enc_np = enc_np[:keep]
                    spec_feats_list.append(enc_np)
                    frames_written_this_file += enc_np.shape[0]
                    total_written += enc_np.shape[0]
                    if max_samples > 0 and total_written >= max_samples:
                        break
            # Normalize this file's progress to [0,1] and align to frames written from this file
            pmin = float(np.min(prog_spec))
            pmax = float(np.max(prog_spec))
            if pmax <= pmin:
                prog_norm_full = np.linspace(0.0, 1.0, num=prog_spec.shape[0], dtype=np.float32)
            else:
                prog_norm_full = ((prog_spec - pmin) / (pmax - pmin)).astype(np.float32)
            if frames_written_this_file > 0:
                spec_prog_list.append(prog_norm_full[:frames_written_this_file])
            if max_samples > 0 and total_written >= max_samples:
                break
        spec_feats = np.concatenate(spec_feats_list, axis=0)
        dir_prog = np.concatenate(spec_prog_list, axis=0) if len(spec_prog_list) > 0 else np.zeros((0,), dtype=np.float32)
        if spec_feats.shape[0] != dir_prog.shape[0]:
            raise ValueError(f"Feature/progress mismatch in dir {dpath.name}: {spec_feats.shape[0]} vs {dir_prog.shape[0]}")
        all_progress.append(dir_prog)
        spec_feats_per_dir.append(spec_feats)
        dir_slices.append((dpath.name, spec_feats.shape[0]))
    all_feats = np.concatenate(spec_feats_per_dir, axis=0)
    all_prog = np.concatenate(all_progress, axis=0).astype(np.float32)
    return all_feats.astype(np.float32), dir_slices, all_prog


def compute_knn_scores(
    base_feats: np.ndarray,
    query_feats: np.ndarray,
    k: int,
    metric: str,
    score_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if base_feats.ndim != 2 or query_feats.ndim != 2:
        raise ValueError('base_feats and query_feats must be 2D arrays [N, D] and [M, D]')
    if base_feats.shape[1] != query_feats.shape[1]:
        raise ValueError(f'Feature dim mismatch: base {base_feats.shape[1]} vs query {query_feats.shape[1]}')
    if k <= 0 or k > base_feats.shape[0]:
        raise ValueError(f'Invalid k={k}; must satisfy 1 <= k <= len(base)={base_feats.shape[0]}')

    # Fit kNN on base and query for distances
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm='auto')
    nn.fit(base_feats)
    dists, _ = nn.kneighbors(query_feats, n_neighbors=k, return_distance=True)

    kth = dists[:, k - 1]
    mean_k = np.mean(dists, axis=1)
    if score_mode == 'kth':
        scores = kth
    elif score_mode == 'mean':
        scores = mean_k
    else:
        raise ValueError(f"Unsupported score mode '{score_mode}'. Use 'mean' or 'kth'.")
    return scores.astype(np.float32), kth.astype(np.float32), mean_k.astype(np.float32)


def compute_train_self_scores(
    base_feats: np.ndarray,
    k: int,
    metric: str,
    score_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Use k+1 neighbors and drop self (first column)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm='auto')
    nn.fit(base_feats)
    dists, _ = nn.kneighbors(base_feats, n_neighbors=k + 1, return_distance=True)
    dists = dists[:, 1:]
    kth = dists[:, k - 1]
    mean_k = np.mean(dists, axis=1)
    if score_mode == 'kth':
        scores = kth
    elif score_mode == 'mean':
        scores = mean_k
    else:
        raise ValueError(f"Unsupported score mode '{score_mode}'. Use 'mean' or 'kth'.")
    return scores.astype(np.float32), kth.astype(np.float32), mean_k.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    # Checkpoints (mirror umap_obs_embeddings.py)
    parser.add_argument('--load_ckpt_file_path', type=str, default=None, help='Explicit path to checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Directory containing checkpoints/ (auto-picks latest.ckpt)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Deprecated alias for --load_ckpt_file_path')

    # Dataset / features
    parser.add_argument('--dataset_path', type=str, default=None, help='Path to training dataset .zarr.zip to embed if features not provided')
    parser.add_argument('--features_path', type=str, default=None, help='Path to precomputed base features .npy (shape [N, D])')
    parser.add_argument('--special_dirs', nargs='*', default=None, help='List of directories containing episode_*.npz as special sets')
    parser.add_argument('--special_features_path', type=str, default=None, help='Path to precomputed special features .npy (shape [M, D])')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')

    # Encoding controls
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--disable_aug', action='store_true', help='Disable vision augmentations for deterministic features')
    parser.add_argument('--max_samples', type=int, default=0, help='Optional limit for debugging (0 = all)')
    parser.add_argument('--eps', type=int, default=0, help='Process only first N episodes from base dataset (0 = all)')
    parser.add_argument('--force_base_encode', action='store_true', help='Force re-encoding base dataset even if out_dir/features.npy exists')

    # kNN / OOD params
    parser.add_argument('--knn_k', type=int, default=50)
    parser.add_argument('--distance_metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    parser.add_argument('--score', type=str, default='mean', choices=['mean', 'kth'])
    parser.add_argument('--calibrate', action='store_true', help='Compute train self kNN distribution and report calibrated scores')
    parser.add_argument('--ood_quantile', type=float, default=0.95, help='Quantile on train scores for OOD threshold when calibrating')

    # Plotting over normalized timesteps
    parser.add_argument('--plot_over_time', action='store_true', help='Generate bucketed score plot over normalized timesteps (requires progress)')
    parser.add_argument('--num_buckets', type=int, default=100, help='Number of buckets over [0,1] for plotting')
    parser.add_argument('--plot_path', type=str, default=None, help='Optional path for the output plot image (PNG). Defaults to out_dir/knn_ood_over_time.png')
    parser.add_argument('--special_progress_path', type=str, default=None, help='Optional npy/npz path with normalized progress [M] aligned to --special_features_path')

    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether encoding is required
    need_base_encode = (args.features_path is None)
    need_special_encode = (args.special_features_path is None)

    # Resolve and load policy only if needed
    policy = None
    cfg = None
    if need_base_encode or need_special_encode:
        resolved_ckpt = resolve_ckpt_path(
            ckpt_path=(args.load_ckpt_file_path or args.ckpt_path),
            ckpt_dir=args.ckpt_dir,
        )
        policy, cfg = load_policy_from_ckpt(resolved_ckpt)

    # Base features (training set)
    if args.features_path is not None:
        base_features = np.load(args.features_path)
    else:
        cache_exists = out_dir.joinpath('features.npy').exists()
        if (not args.force_base_encode) and cache_exists:
            base_features = np.load(out_dir.joinpath('features.npy'))
        else:
            if args.dataset_path is None:
                raise ValueError('Either --features_path or --dataset_path must be provided for base set')
            base_features = encode_base_features(
                policy=policy,
                cfg=cfg,
                dataset_path=args.dataset_path,
                out_dir=out_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                disable_aug=args.disable_aug,
                max_samples=args.max_samples,
                eps=args.eps,
            )

    if base_features.dtype != np.float32:
        base_features = base_features.astype(np.float32)

    # Special features (query set)
    special_progress: Optional[np.ndarray] = None
    if args.special_features_path is not None:
        special_features = np.load(args.special_features_path)
        dir_slices: List[Tuple[str, int]] = [('special', int(special_features.shape[0]))]
        if args.special_progress_path is not None:
            # support npz with array under key 'progress' or bare npy
            if args.special_progress_path.endswith('.npz'):
                data = np.load(args.special_progress_path)
                if 'progress' in data:
                    special_progress = data['progress'].astype(np.float32)
                else:
                    keys = list(data.keys())
                    raise KeyError(f"Progress npz must contain 'progress' array. Found keys: {keys}")
            else:
                special_progress = np.load(args.special_progress_path).astype(np.float32)
    else:
        if args.special_dirs is None or len(args.special_dirs) == 0:
            raise ValueError('Provide either --special_features_path or at least one directory in --special_dirs')
        special_features, dir_slices, special_progress = encode_special_dirs_features(
            policy=policy,
            cfg=cfg,
            special_dirs=args.special_dirs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

    if special_features.dtype != np.float32:
        special_features = special_features.astype(np.float32)

    # Compute kNN-based OOD scores
    scores_raw, kth_d, mean_d = compute_knn_scores(
        base_feats=base_features,
        query_feats=special_features,
        k=args.knn_k,
        metric=args.distance_metric,
        score_mode=args.score,
    )

    # Optional calibration
    calibrated_z = None
    calibrated_percentile = None
    threshold = None
    is_ood = None
    if args.calibrate:
        train_scores, _tk, _tm = compute_train_self_scores(
            base_feats=base_features,
            k=args.knn_k,
            metric=args.distance_metric,
            score_mode=args.score,
        )
        mu = float(np.mean(train_scores))
        sigma = float(np.std(train_scores))
        if sigma <= 0.0:
            raise ValueError('Train self score std is zero; calibration not meaningful')
        calibrated_z = (scores_raw - mu) / sigma
        # empirical percentile
        calibrated_percentile = np.searchsorted(np.sort(train_scores), scores_raw, side='right') / train_scores.shape[0]
        threshold = float(np.quantile(train_scores, args.ood_quantile))
        is_ood = (scores_raw > threshold)

        # Save train self distribution for reuse
        np.save(out_dir.joinpath('train_knn_self_scores.npy'), train_scores)

    # Write per-sample CSV
    import csv
    csv_path = out_dir.joinpath('knn_ood.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['dir_name', 'index_in_dir', 'score_raw', 'kth_distance', 'mean_k_distance']
        if args.calibrate:
            header += ['z_score', 'percentile', 'is_ood']
        writer.writerow(header)
        offset = 0
        for dir_name, count in dir_slices:
            for i in range(count):
                row = [dir_name, i, float(scores_raw[offset + i]), float(kth_d[offset + i]), float(mean_d[offset + i])]
                if args.calibrate:
                    row += [float(calibrated_z[offset + i]), float(calibrated_percentile[offset + i]), bool(is_ood[offset + i])]
                writer.writerow(row)
            offset += count

    # Write summary JSON
    summary = {
        'num_base': int(base_features.shape[0]),
        'num_query': int(special_features.shape[0]),
        'k': int(args.knn_k),
        'metric': args.distance_metric,
        'score_mode': args.score,
        'dirs': [{'name': n, 'count': int(c)} for (n, c) in dir_slices],
    }
    if args.calibrate:
        summary.update({
            'ood_quantile': float(args.ood_quantile),
            'threshold': float(threshold),
        })
    with open(out_dir.joinpath('knn_ood.json'), 'w') as jf:
        json.dump(summary, jf, indent=2)

    print(f"Saved per-sample scores to: {str(csv_path)}")
    print(f"Saved summary to: {str(out_dir.joinpath('knn_ood.json'))}")

    # Optional: plot over normalized timesteps
    if args.plot_over_time:
        if special_progress is None:
            raise ValueError('--plot_over_time requires progress for special set. Provide --special_dirs (with episode_*.npz) or --special_progress_path aligned to --special_features_path')
        if special_progress.shape[0] != special_features.shape[0]:
            raise ValueError(f'Progress length mismatch: progress {special_progress.shape[0]} vs features {special_features.shape[0]}')
        num_buckets = int(args.num_buckets)
        if num_buckets <= 1:
            raise ValueError('--num_buckets must be > 1')
        prog = np.clip(special_progress.reshape(-1), 0.0, 1.0)
        bucket_idx = np.minimum((prog * num_buckets).astype(int), num_buckets - 1)
        # aggregate
        mean_kth_by_bucket = np.full(num_buckets, np.nan, dtype=np.float32)
        mean_mean_by_bucket = np.full(num_buckets, np.nan, dtype=np.float32)
        counts = np.zeros(num_buckets, dtype=np.int64)
        for b in range(num_buckets):
            mask = (bucket_idx == b)
            counts[b] = int(np.sum(mask))
            if counts[b] > 0:
                mean_kth_by_bucket[b] = float(np.mean(kth_d[mask]))
                mean_mean_by_bucket[b] = float(np.mean(mean_d[mask]))

        # global stats
        g_mean_kth = float(np.mean(kth_d))
        g_std_kth = float(np.std(kth_d))
        g_mean_mean = float(np.mean(mean_d))
        g_std_mean = float(np.std(mean_d))

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        xs = (np.arange(num_buckets) + 0.5) / num_buckets
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xs, mean_kth_by_bucket, label='k-th neighbor distance', color='#E67E22', linewidth=2)
        ax.plot(xs, mean_mean_by_bucket, label='mean of k distances', color='#2980B9', linewidth=2)

        # horizontal global means with std bands
        ax.axhline(g_mean_kth, color='#E67E22', linestyle='--', linewidth=1, alpha=0.8, label='k-th global mean')
        ax.fill_between(xs, g_mean_kth - g_std_kth, g_mean_kth + g_std_kth, color='#E67E22', alpha=0.12)
        ax.axhline(g_mean_mean, color='#2980B9', linestyle='--', linewidth=1, alpha=0.8, label='mean-k global mean')
        ax.fill_between(xs, g_mean_mean - g_std_mean, g_mean_mean + g_std_mean, color='#2980B9', alpha=0.12)

        ax.set_xlabel('Normalized episode progress')
        ax.set_ylabel('Distance')
        ax.set_title(f'kNN OOD score over normalized timesteps (k={args.knn_k}, metric={args.distance_metric})')
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
        ax.legend()
        plot_path = pathlib.Path(args.plot_path) if args.plot_path is not None else out_dir.joinpath('knn_ood_over_time.png')
        # Annotate means/stds in-figure
        annotation = (
            f"k-th: mean={g_mean_kth:.4f}, std={g_std_kth:.4f}\n"
            f"mean-k: mean={g_mean_mean:.4f}, std={g_std_mean:.4f}"
        )
        ax.text(
            0.98,
            0.02,
            annotation,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
        fig.tight_layout()
        fig.savefig(str(plot_path), dpi=180)
        plt.close(fig)

        # Also export buckets CSV
        buckets_csv = out_dir.joinpath('knn_ood_buckets.csv')
        import csv as _csv
        with open(buckets_csv, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['bucket', 'center', 'count', 'mean_kth_distance', 'mean_mean_distance'])
            for b in range(num_buckets):
                w.writerow([b, float(xs[b]), int(counts[b]), float(mean_kth_by_bucket[b]) if not np.isnan(mean_kth_by_bucket[b]) else '', float(mean_mean_by_bucket[b]) if not np.isnan(mean_mean_by_bucket[b]) else ''])
        print(f"Saved plot to: {str(plot_path)}")
        print(f"Saved bucketed stats to: {str(buckets_csv)}")


if __name__ == '__main__':
    main()


