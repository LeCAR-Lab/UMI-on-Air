import os
import sys
import argparse
import pathlib
import dill
import numpy as np
import torch
import plotly.graph_objects as go

# Ensure imports resolve when running this script directly
UI_DIR = pathlib.Path(__file__).parent.parent  # .../universal_manipulation_interface
REPO_DIR = UI_DIR.parent                       # .../am_mujoco_ws
for p in [str(UI_DIR), str(REPO_DIR)]:
    if p not in sys.path:
        sys.path.append(p)
os.chdir(str(UI_DIR))

from omegaconf import OmegaConf
import hydra

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.umi_dataset import UmiDataset
from tqdm import tqdm
from diffusion_policy.common.sampler import SequenceSampler


def resolve_plotly_colorscale(name: str) -> str:
    """Map a common matplotlib-style name to a Plotly colorscale name.
    Raises ValueError if not recognized (no silent fallback).
    """
    name_lower = (name or '').lower()
    mapping = {
        'plasma': 'Plasma',
        'viridis': 'Viridis',
        'inferno': 'Inferno',
        'magma': 'Magma',
        'cividis': 'Cividis',
        'turbo': 'Turbo',
        'spectral': 'Spectral',  # diverging
        'greys': 'Greys',
        'gray': 'Greys',
        'grey': 'Greys',
    }
    if name_lower in mapping:
        return mapping[name_lower]
    raise ValueError(f"Unsupported colorscale '{name}'. Choose one of: {sorted(mapping.keys())}")


def choose_contrasting_colorscale(base_colorscale_name: str) -> str:
    """Return a contrasting Plotly colorscale for the given Plotly colorscale name.
    Input should be a Plotly name (case-insensitive), e.g., 'Plasma'.
    """
    base_lower = (base_colorscale_name or '').lower()
    mapping = {
        'plasma': 'Viridis',
        'viridis': 'Inferno',
        'inferno': 'Cividis',
        'magma': 'Turbo',
        'cividis': 'Plasma',
        'turbo': 'Cividis',
        'spectral': 'Plasma',
        'greys': 'Plasma',
    }
    return mapping.get(base_lower, 'Inferno')


def resolve_ckpt_path(ckpt_path: str = None, ckpt_dir: str = None):
    if ckpt_path is not None:
        return ckpt_path
    if ckpt_dir is not None:
        latest = pathlib.Path(ckpt_dir).joinpath('checkpoints', 'latest.ckpt')
        if latest.exists():
            return str(latest)
        # fallback to any .ckpt in checkpoints
        ckpt_dir_path = pathlib.Path(ckpt_dir).joinpath('checkpoints')
        if ckpt_dir_path.exists():
            all_ckpts = sorted(ckpt_dir_path.glob('*.ckpt'))
            if len(all_ckpts) > 0:
                return str(all_ckpts[-1])
    raise FileNotFoundError('Could not resolve checkpoint path. Provide --load_ckpt_file_path or --ckpt_dir.')


def load_policy_from_ckpt(ckpt_path: str):
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    # Recreate workspace and load payload exactly as training
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


def build_dataset_from_cfg(cfg, val_ratio: float = 0.0, dataset_path_override: str = None):
    # Pull task-level shape_meta and dataset_path
    shape_meta = cfg.task.shape_meta
    dataset_path = dataset_path_override if dataset_path_override is not None else cfg.task.dataset_path
    pose_repr = cfg.task.pose_repr

    # Use the same horizons/downsampling embedded in shape_meta
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


def compute_episode_progress_indices(dataset: UmiDataset) -> np.ndarray:
    # dataset.sampler.indices: list of tuples (current_idx, start_idx, end_idx, before_first_grasp)
    progresses = []
    for (current_idx, start_idx, end_idx, _flag) in dataset.sampler.indices:
        length = max(1, end_idx - start_idx)
        p = (current_idx - start_idx) / (length - 1) if length > 1 else 0.0
        p = float(np.clip(p, 0.0, 1.0))
        progresses.append(p)
    return np.asarray(progresses, dtype=np.float32)


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


def build_progress_colors(progress: np.ndarray, cmap_name: str = 'plasma') -> np.ndarray:
    # Deprecated: kept for compatibility; not used with Plotly rendering
    raise NotImplementedError('build_progress_colors is not used; Plotly uses numeric progress with a colorscale.')


def main():
    parser = argparse.ArgumentParser()
    # Checkpoint resolution (imitate_episodes-style)
    parser.add_argument('--load_ckpt_file_path', type=str, default=None, help='Explicit path to checkpoint file')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Directory containing checkpoints/ (auto-picks latest.ckpt)')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Deprecated alias for --load_ckpt_file_path')

    # Dataset override
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset .zarr.zip to embed')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--disable_aug', action='store_true', help='Disable vision augmentations for deterministic features')
    parser.add_argument('--max_samples', type=int, default=0, help='Optional limit for debugging (0 = all)')
    parser.add_argument('--umap_n_neighbors', type=int, default=50)
    parser.add_argument('--umap_min_dist', type=float, default=0.25)
    parser.add_argument('--colormap', type=str, default='viridis', help='Colorscale for base progress (e.g., plasma)')
    parser.add_argument('--special_colormap', type=str, default=None, help='Colorscale for all special dirs (defaults to a contrasting palette)')
    parser.add_argument('--special_colormaps', nargs='*', default=None, help='Per-dir colorscales for specials (one per entry in --special_dirs)')
    parser.add_argument('--eps', type=int, default=0, help='Process only first N episodes (0 = all)')
    parser.add_argument('--special_dirs', nargs='*', default=None, help='List of directories containing episode_*.npz to overlay as special sets')
    parser.add_argument('--force_base_encode', action='store_true', help='Force re-encoding base dataset even if cached features.npy exists in out_dir')
    parser.add_argument('--special_only', action='store_true', help='Fit and plot only special dirs; exclude base dataset entirely')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve and load policy + cfg
    resolved_ckpt = resolve_ckpt_path(
        ckpt_path=(args.load_ckpt_file_path or args.ckpt_path),
        ckpt_dir=args.ckpt_dir,
    )
    policy, cfg = load_policy_from_ckpt(resolved_ckpt)

    # Base features: optionally skip entirely when special_only is set
    base_features = None
    base_progress = None
    total = 0
    if not args.special_only:
        if (not args.force_base_encode) and out_dir.joinpath('features.npy').exists() and out_dir.joinpath('progress.npy').exists():
            base_features = np.load(out_dir.joinpath('features.npy'))
            base_progress = np.load(out_dir.joinpath('progress.npy'))
            total = base_features.shape[0]
        else:
            # Instantiate full dataset (all episodes), overriding dataset path as requested
            dataset = build_dataset_from_cfg(cfg, val_ratio=0.0, dataset_path_override=args.dataset_path)
            if args.eps and args.eps > 0:
                restrict_dataset_to_first_eps(dataset, args.eps)
            # DataLoader in dataset order
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            # Optionally disable augmentation by replacing transforms with Identity
            if args.disable_aug and hasattr(policy, 'obs_encoder'):
                import torch.nn as nn
                if hasattr(policy.obs_encoder, 'key_transform_map'):
                    for k in list(policy.obs_encoder.key_transform_map.keys()):
                        policy.obs_encoder.key_transform_map[k] = nn.Identity()
            # Build per-sample progress
            base_progress = compute_episode_progress_indices(dataset)
            # Dry run to determine feature dimension
            with torch.no_grad():
                first_batch = next(iter(loader))
                obs = first_batch['obs']
                device = policy.device
                obs = dict_apply(obs, lambda x: x.to(device=device, non_blocking=True))
                nobs = policy.normalizer.normalize(obs)
                feats = policy.obs_encoder(nobs)
                feat_dim = int(np.prod(list(feats.shape[1:])))
            # Encode base
            N = len(dataset)
            features_path = out_dir.joinpath('features_memmap.float32')
            features = np.memmap(features_path, mode='w+', dtype=np.float32, shape=(N, feat_dim))
            idx_base = 0
            with torch.no_grad():
                for batch in tqdm(loader, desc='Encoding features', unit='batch'):
                    obs = batch['obs']
                    B = next(iter(obs.values())).shape[0]
                    if args.max_samples > 0 and (idx_base >= args.max_samples):
                        break
                    obs = dict_apply(obs, lambda x: x.to(device=policy.device, non_blocking=True))
                    nobs = policy.normalizer.normalize(obs)
                    enc = policy.obs_encoder(nobs)
                    enc_np = enc.detach().cpu().numpy().reshape(B, -1).astype(np.float32)
                    write_end = idx_base + B
                    if args.max_samples > 0:
                        write_end = min(write_end, args.max_samples)
                        enc_np = enc_np[: write_end - idx_base]
                    features[idx_base:write_end] = enc_np
                    idx_base = write_end
                    if args.max_samples > 0 and idx_base >= args.max_samples:
                        break
            total = idx_base
            features.flush()
            base_features = np.array(features[:total])
            # Cache base
            np.save(out_dir.joinpath('features.npy'), base_features)
            np.save(out_dir.joinpath('progress.npy'), base_progress[:total])

    # Load only the filled portion for UMAP (if using base)
    feats_in = base_features if (not args.special_only) else None

    # Optionally load and encode special DIRs first, so we can refit on all data by default
    special_dir_names = []
    special_embeds_slices = []  # list of (name, count) to slice after fit
    special_prog_per_dir = []
    spec_feats_all = None
    if args.special_dirs is not None and len(args.special_dirs) > 0:
        spec_feats_dir_list = []
        for dir_idx, dir_path in enumerate(args.special_dirs):
            dpath = pathlib.Path(dir_path)
            npz_files = sorted(dpath.glob('episode_*.npz'))
            if len(npz_files) == 0:
                raise FileNotFoundError(f'No episode_*.npz found in {dpath}')
            # encode all files in this dir
            spec_feats_list = []
            spec_prog_list = []
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
                bs = max(1, min(args.batch_size, Btot))
                # Build full obs dict expected by encoder; zero-fill any missing low-dim keys to keep feature dim consistent
                shape_meta = cfg.task.shape_meta['obs']
                expected_lowdim = list(getattr(policy.obs_encoder, 'low_dim_keys', []))
                with torch.no_grad():
                    for start in range(0, Btot, bs):
                        end = min(start + bs, Btot)
                        obs_chunk = {
                            'camera0_rgb': torch.from_numpy(obs_np['camera0_rgb'][start:end]).float().to(policy.device)
                        }
                        for key in expected_lowdim:
                            if key in obs_np:
                                obs_chunk[key] = torch.from_numpy(obs_np[key][start:end]).float().to(policy.device)
                            else:
                                # zero-fill with correct (T, D)
                                T = int(shape_meta[key]['horizon'])
                                D = int(shape_meta[key]['shape'][-1])
                                obs_chunk[key] = torch.zeros((end - start, T, D), dtype=torch.float32, device=policy.device)
                        nobs = policy.normalizer.normalize(obs_chunk)
                        enc = policy.obs_encoder(nobs)
                        spec_feats_list.append(enc.detach().cpu().numpy().reshape(end - start, -1))
                spec_prog_list.append(prog_spec)
            spec_feats = np.concatenate(spec_feats_list, axis=0)
            spec_prog = np.concatenate(spec_prog_list, axis=0)
            spec_feats_dir_list.append(spec_feats)
            # store per-dir progress for Plotly color mapping
            special_prog_per_dir.append(spec_prog)
            special_dir_names.append(dpath.name)
        # concat all specials features and remember slices
        counts = [arr.shape[0] for arr in spec_feats_dir_list]
        spec_feats_all = np.concatenate(spec_feats_dir_list, axis=0)
        special_embeds_slices = list(zip(special_dir_names, counts))

    # UMAP embedding (always refit fresh on base + specials if provided)
    from umap import UMAP
    reducer = UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric='euclidean',
        random_state=42,
    )
    num_base = 0 if args.special_only or feats_in is None else feats_in.shape[0]
    num_special = 0 if spec_feats_all is None else spec_feats_all.shape[0]
    num_samples = num_base + num_special
    print(f"Fitting UMAP (2D) on {num_samples} samples; embeddings are not loaded from or saved to disk.")
    if args.special_only:
        if spec_feats_all is None:
            raise ValueError("--special_only requires at least one directory in --special_dirs")
        embedding = reducer.fit_transform(spec_feats_all)
    else:
        if spec_feats_all is not None:
            combined = np.concatenate([feats_in, spec_feats_all], axis=0)
            embedding = reducer.fit_transform(combined)
        else:
            embedding = reducer.fit_transform(feats_in)

    # Plot 2D with Plotly (interactive HTML)
    colorscale = resolve_plotly_colorscale(args.colormap)
    # Determine colorscales for specials: per-dir if provided, else a single contrasting scale
    special_colorscales: list[str] = []
    if args.special_colormaps is not None and len(args.special_colormaps) > 0:
        special_colorscales = [resolve_plotly_colorscale(cs) for cs in args.special_colormaps]
    elif args.special_colormap is not None:
        special_colorscales = [resolve_plotly_colorscale(args.special_colormap)]
    else:
        special_colorscales = []  # will auto-assign contrasting in loop
    fig = go.Figure()
    print(f"Base colorscale: {colorscale}")
    if len(special_colorscales) == 1:
        print(f"Special colorscale (all dirs): {special_colorscales[0]}")

    # Base points (only when not special_only)
    if not args.special_only and total > 0:
        base_embed = embedding[:total]
        base_prog_plot = np.clip(base_progress[:total].astype(float), 0.0, 1.0)
        print(f"Base progress range (clipped): [{base_prog_plot.min():.3f}, {base_prog_plot.max():.3f}]")
        fig.add_trace(
            go.Scatter(
                x=base_embed[:, 0],
                y=base_embed[:, 1],
                mode='markers',
                name='base',
                marker=dict(
                    size=4,
                    color=base_prog_plot,
                    colorscale=colorscale,
                    cmin=0.0,
                    cmax=1.0,
                    showscale=True,
                    colorbar=dict(title='progress'),
                    line=dict(width=0),
                ),
                opacity=0.85,
            )
        )

    # Overlay specials (already included in fit)
    if spec_feats_all is not None:
        offset = 0 if args.special_only else total
        for idx, ((dir_name, count), prog_dir) in enumerate(zip(special_embeds_slices, special_prog_per_dir)):
            spec_embed_dir = embedding[offset:offset+count]
            assert len(prog_dir) == count, f"Progress length mismatch for {dir_name}: {len(prog_dir)} != {count}"
            prog_dir = prog_dir.astype(float).reshape(-1)
            # Normalize per-dir to [0,1] to fully span the colorscale regardless of absolute values
            pmin = float(np.min(prog_dir))
            pmax = float(np.max(prog_dir))
            if pmax <= pmin:
                norm_prog = np.linspace(0.0, 1.0, num=count, dtype=float)
            else:
                norm_prog = (prog_dir - pmin) / (pmax - pmin)
            norm_prog = np.clip(norm_prog, 0.0, 1.0)
            print(f"Special '{dir_name}' progress normalized to [0,1]; raw range was [{pmin:.3f}, {pmax:.3f}]")
            # pick per-dir colorscale
            if len(special_colorscales) >= (idx + 1):
                dir_colorscale = special_colorscales[idx]
            elif len(special_colorscales) == 1:
                dir_colorscale = special_colorscales[0]
            else:
                # auto-assign contrasting by cycling several palettes
                auto_cycle = ['Inferno', 'Turbo', 'Cividis', 'Plasma', 'Viridis', 'Magma', 'Spectral']
                dir_colorscale = auto_cycle[idx % len(auto_cycle)]
            print(f"Special '{dir_name}' colorscale: {dir_colorscale}")
            fig.add_trace(
                go.Scatter(
                    x=spec_embed_dir[:, 0],
                    y=spec_embed_dir[:, 1],
                    mode='markers',
                    name=dir_name,
                    marker=dict(
                        size=6,
                        color=norm_prog,
                        colorscale=dir_colorscale,
                        cmin=0.0,
                        cmax=1.0,
                        showscale=(args.special_only and idx == 0),
                        colorbar=(dict(title='progress') if (args.special_only and idx == 0) else None),
                        line=dict(width=0),
                    ),
                    opacity=0.95,
                )
            )
            offset += count

    fig.update_layout(
        title='UMAP of Observations (colored by episode progress)',
        xaxis_title='UMAP-1',
        yaxis_title='UMAP-2',
        template='plotly_white',
        legend=dict(itemsizing='trace')
    )
    png_path = out_dir.joinpath('umap_2d.png')
    # Requires 'kaleido' to be installed for static image export
    fig.write_image(str(png_path), format='png', scale=2)
    print(f"Saved UMAP 2D PNG to: {str(png_path)}")


if __name__ == '__main__':
    # Ensure Hydra resolvers available (as in train.py / imitate_episodes)
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    main()


