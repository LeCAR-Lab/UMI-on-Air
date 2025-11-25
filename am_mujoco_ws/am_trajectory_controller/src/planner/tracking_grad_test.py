#!/usr/bin/env python3
"""
tracking_grad_test.py
---------------------
Small standalone script that checks the analytical gradient of the
tracking-only cost returned by `DroneMPC_4DOF.get_tracking_gradients()`
against a finite-difference approximation.

Usage
-----
$ python tracking_grad_test.py path/to/ee_mpc_test.yaml

The YAML file must follow the same structure used by `ee_mpc_test.py`:
  • It contains a key `mpc_config_path` that points to a second YAML file
    with all MPC parameters.
  • The trajectory CSV referenced inside is used to build the reference
    horizon.

The script perturbs **only** the end-effector position references (x,y,z)
independently at every knot over the horizon, computes the new tracking
cost, and compares the centred finite difference with the analytical
value.  A summary of max and mean relative errors is printed.

This does NOT modify the solver or regenerate any code; it merely calls
`run_optimization_with_tracking_cost` multiple times.
"""
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import tempfile, csv
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

from ee_mpc_acado_4dof import ArmMPCPlanner
from ee_mpc_test import DroneSimEnv, xyzw_to_wxyz, quaternion_to_rpy  # reuse helpers

# --------------------------------------------------------------------------------------
EPS = 1e-6          # finite-difference step in metres
REL_TOL = 1e-4      # acceptable relative error


def main(test_yaml_path: str):
    # ------------------------------------------------------------------
    # Build the simulation environment just to reuse loading utilities
    # (planner + reference trajectory). We will not run the full sim.
    # ------------------------------------------------------------------
    env = DroneSimEnv(test_yaml_path)
    planner = env.planner  # ArmMPCPlanner instance

    # Grab initial state (reuse the setup logic of DroneSimEnv)
    ref0 = env.ref_traj[0]
    ee_pos_ref0 = ref0[:3]
    ee_quat_ref0 = ref0[3:7]
    ee_quat_ref0_wxyz = xyzw_to_wxyz(ee_quat_ref0.reshape(1, 4))[0]
    ee_euler_ref0 = quaternion_to_rpy(ee_quat_ref0_wxyz)  # degrees = False
    base_yaw = ee_euler_ref0[2]

    # initial guesses
    p = np.zeros(3)
    v = np.zeros(3)
    base_euler = np.array([0.0, 0.0, base_yaw])
    arm_angle = np.array([0.0, 0.0, 0.0, 90.0]) / 180 * np.pi
    arm_angle[0] = ee_euler_ref0[1]

    # Shift base position so that the forward-kinematics EE pose matches
    # the very first reference exactly (important for clear plotting).
    ee_state_init = planner.forward_kinematics(p, base_euler, arm_angle)
    ee_init_pos = ee_state_init[:3]
    ee_offset = ee_pos_ref0 - ee_init_pos  # desired minus current
    p = p + ee_offset

    # Previous control initialisation (after p is finalised)
    u_prev = np.zeros(10)
    u_prev[6:] = arm_angle

    # Horizon references (positions & quaternions)
    horizon = planner.mpc.N
    ee_state_refs = env.ref_traj[: horizon + 1]
    ee_pos_refs = ee_state_refs[:, :3]
    ee_quat_refs = xyzw_to_wxyz(ee_state_refs[:, 3:7])

    # Solve once to obtain analytical gradient and store state/control trajectory
    planner.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
    x0 = np.concatenate([p, v, base_euler, np.zeros(3), arm_angle])
    # Capture optimizer states to later compute actual EE trajectory
    _, x_opt_orig, total_cost_orig, _ = planner.mpc.run_optimization_with_tracking_cost(x0)

    g_exact = planner.mpc.get_tracking_gradients()

    # ------------------------------------------------------------------
    # Helper to compute world-frame EE positions from optimizer states
    # ------------------------------------------------------------------
    def compute_ee_positions(x_seq: np.ndarray) -> np.ndarray:
        pos_list = []
        for xk in x_seq:
            p_k = xk[0:3]
            base_euler_k = xk[6:9]
            arm_angle_k = xk[12:16]
            ee_state_k = planner.forward_kinematics(p_k, base_euler_k, arm_angle_k)
            pos_list.append(ee_state_k[:3])
        return np.array(pos_list)

    # Utility to evaluate tracking cost for a custom set of EE position refs
    def eval_tracking_cost(pos_refs_custom: np.ndarray) -> float:
        dt = planner.mpc.T / planner.mpc.N
        J = 0.0
        # NOTE: Only summing stage costs, ignoring terminal cost
        for k in range(planner.mpc.N):
            xk = planner.mpc.ocp_solver.get(k, "x")
            uk = planner.mpc.ocp_solver.get(k, "u")
            pk = planner.mpc.ocp_solver.get(k, "p").copy()
            pk[0:3] = pos_refs_custom[k]
            J += dt * float(planner.mpc._trk_stage_cost_func(xk, uk, pk))
        return J

    # The full reference trajectory has N+1 points, but for this test we only
    # care about the N points corresponding to the MPC stages.
    ee_pos_refs_stages = ee_pos_refs[:planner.mpc.N].copy()

    J_nom = eval_tracking_cost(ee_pos_refs_stages)

    print(f"\nOriginal costs:")
    print(f"  Total optimization cost: {float(total_cost_orig):.6e}")
    print(f"  Tracking cost only: {J_nom:.6e}")

    # allocate finite-difference gradient
    g_fd = np.zeros_like(g_exact)

    N_stages = planner.mpc.N

    print(f"Computing finite-difference gradients for {N_stages} stages (no re-solve) ...")
    iterator = tqdm.tqdm(range(N_stages * 3))
    for flat_idx in iterator:
        k = flat_idx // 3
        dim = flat_idx % 3
        iterator.set_description(f"perturb k={k} dim={dim}")

        pos_refs_plus = ee_pos_refs_stages.copy()
        pos_refs_plus[k, dim] += EPS
        J_plus = eval_tracking_cost(pos_refs_plus)

        pos_refs_minus = ee_pos_refs_stages.copy()
        pos_refs_minus[k, dim] -= EPS
        J_minus = eval_tracking_cost(pos_refs_minus)

        g_fd[flat_idx] = (J_plus - J_minus) / (2 * EPS)

    # Compare
    abs_err = np.abs(g_exact - g_fd)
    rel_err = abs_err / np.maximum(1.0, np.abs(g_exact))

    print("\nGradient check summary:")
    print(f"  max |abs error| : {abs_err.max():.3e}")
    print(f"  max |rel error| : {rel_err.max():.3e}")
    print(f"  mean|rel error| : {rel_err.mean():.3e}")

    if rel_err.max() < REL_TOL:
        print("\n✔ Gradients match finite-difference approximation within tolerance.")
    else:
        bad = np.where(rel_err > REL_TOL)[0]
        print(f"\n❌ {bad.size} entries exceed tolerance. Inspect rel_err array.")

    # --------------------------------------------------------------
    # Visualise effect of one gradient-descent step on EE references
    # --------------------------------------------------------------
    step_size = 0.1  # metres per unit gradient
    g_mat = g_exact.reshape(N_stages, 3)
    ee_pos_refs_updated = ee_pos_refs_stages - step_size * g_mat

    J_updated = eval_tracking_cost(ee_pos_refs_updated)
    print(f"\nTracking cost after one gradient-descent step (step_size={step_size:.3f}): {J_updated:.6e}")
    print(f"Cost reduction: {J_nom - J_updated:.6e}")

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('EE position references before and after one gradient step')
    
    knot_indices = np.arange(N_stages)
    axis_labels = ['x', 'y', 'z']

    for i, label in enumerate(axis_labels):
        axs[i].plot(knot_indices, ee_pos_refs_stages[:, i], 'bo-', label='original')
        axs[i].plot(knot_indices, ee_pos_refs_updated[:, i], 'ro-', label='after grad step')
        axs[i].set_ylabel(f'{label} [m]')
        axs[i].legend()
        axs[i].grid(True)
    
    axs[-1].set_xlabel('Knot index')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = Path("tracking_gradient_update.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path.resolve()}")

    # ------------------------------------------------------------------
    # Compute and plot actual EE trajectories (original vs updated refs)
    # ------------------------------------------------------------------
    # Actual EE trajectory for original references (already solved)
    ee_pos_actual_orig = compute_ee_positions(x_opt_orig)

    # Re-solve MPC with updated position references to obtain new trajectory
    ee_pos_refs_updated_full = np.vstack((ee_pos_refs_updated, ee_pos_refs_updated[-1:]))  # add terminal point
    planner.mpc.set_reference_sequence(ee_pos_refs_updated_full, ee_quat_refs, arm_angle, u_prev)
    _, x_opt_updated, total_cost_updated = planner.mpc.run_optimization(x0)
    ee_pos_actual_updated = compute_ee_positions(x_opt_updated)

    print(f"\nUpdated costs after re-optimization:")
    print(f"  Total optimization cost: {float(total_cost_updated):.6e}")
    print(f"  Total cost reduction: {float(total_cost_orig) - float(total_cost_updated):.6e}")

    # Prepare reference sequences for plotting (full N+1 length)
    ee_pos_refs_full = ee_pos_refs  # original full reference (N+1)

    # 3D plot of trajectories (cleaner visuals)
    fig_traj = plt.figure(figsize=(8, 6))
    ax3d = fig_traj.add_subplot(111, projection='3d')
    ax3d.plot(ee_pos_refs_full[:, 0], ee_pos_refs_full[:, 1], ee_pos_refs_full[:, 2], color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='Ref Orig')
    ax3d.plot(ee_pos_refs_updated_full[:, 0], ee_pos_refs_updated_full[:, 1], ee_pos_refs_updated_full[:, 2], color='green', linestyle='--', linewidth=1.0, alpha=0.7, label='Ref Upd')

    ax3d.plot(ee_pos_actual_orig[:, 0], ee_pos_actual_orig[:, 1], ee_pos_actual_orig[:, 2], 'b-', linewidth=2.0, label='Actual Orig')
    ax3d.plot(ee_pos_actual_updated[:, 0], ee_pos_actual_updated[:, 1], ee_pos_actual_updated[:, 2], 'r-', linewidth=2.0, label='Actual Upd')

    # Mark start
    ax3d.scatter(0, 0, 0, c='k', marker='x', s=40, label='Start')

    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.set_title('EE Trajectories (3-D view)')
    ax3d.legend()
    fig_traj.tight_layout()
    traj_path = Path('actual_ee_trajectories_3d.png')
    fig_traj.savefig(traj_path)
    print(f"3D trajectory plot saved to {traj_path.resolve()}")

    # --------------------------------------------------------------
    # Component-wise plots for clearer inspection
    # --------------------------------------------------------------
    # Ensure all series share the same length (clip to the shortest)
    min_len = min(ee_pos_refs_full.shape[0],
                  ee_pos_refs_updated_full.shape[0],
                  ee_pos_actual_orig.shape[0],
                  ee_pos_actual_updated.shape[0])

    comp_indices = np.arange(min_len)
    fig_comp, axs_comp = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels_xyz = ['x', 'y', 'z']
    for i, lbl in enumerate(labels_xyz):
        axs_comp[i].plot(comp_indices, ee_pos_refs_full[:min_len, i], linestyle='--', color='gray', label='Ref Orig')
        axs_comp[i].plot(comp_indices, ee_pos_refs_updated_full[:min_len, i], linestyle='--', color='green', label='Ref Upd')
        axs_comp[i].plot(comp_indices, ee_pos_actual_orig[:min_len, i], color='blue', label='Actual Orig')
        axs_comp[i].plot(comp_indices, ee_pos_actual_updated[:min_len, i], color='red', label='Actual Upd')
        axs_comp[i].set_ylabel(f'{lbl} [m]')
        axs_comp[i].grid(True, alpha=0.3)
        axs_comp[i].legend(loc='best')

    axs_comp[-1].set_xlabel('Knot index')
    fig_comp.suptitle('End-Effector Trajectories – Components')
    fig_comp.tight_layout(rect=[0, 0.03, 1, 0.97])
    comp_path = Path('actual_ee_trajectories_components.png')
    fig_comp.savefig(comp_path)
    plt.close(fig_comp)  # ensure file is flushed & closed
    print(f"Component-wise trajectory plot saved to {comp_path.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        test_yaml = sys.argv[1]
    else:
        # ------------------------------------------------------------------
        # Auto-generate a demo trajectory & YAML config if none provided
        # ------------------------------------------------------------------
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)

        # Determine required trajectory length from MPC config (N_mini + 1)
        config_path = (Path(__file__).parent / "../../config/traj_sender_mpc_ee.yaml").resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Expected MPC config at {config_path} but not found.")
        with open(config_path, 'r') as f:
            mpc_config = yaml.safe_load(f)
        n_points = mpc_config['mpc']['N_mini']

        # Build a trajectory with n_points knots
        horizon_N = mpc_config['mpc']['N_mini']
        print(f"Auto-generating a {n_points}-point trajectory for MPC horizon N={horizon_N}")
        x_vals = np.linspace(0.0, 0.1, n_points)
        z_arc  = np.sin(np.linspace(0, np.pi, n_points)) * 0.1 # smooth arch
        y_vals = np.zeros(n_points)
        times  = np.arange(n_points) * 0.08  # 80 ms increments

        csv_path = tmp_path / "arch8_traj.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "px", "py", "pz", "qx", "qy", "qz", "qw"])
            for t, x, y, z in zip(times, x_vals, y_vals, z_arc):
                writer.writerow([f"{t:.3f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", 0.0, 0.0, 0.0, 1.0])

        # YAML pointing to existing MPC config & the csv we just made
        yaml_path = tmp_path / "arch8_test.yaml"
        test_yaml_dict = {
            "mpc_config_path": str(config_path),
            "trajectory": {
                "csv_path": str(csv_path)
            },
            "simulation": {
                "dt": 0.08,
                "acc2thrust_gain": [0.1, 0.12, 0.1],
                "thrust_bias": [0.005, 0.005, 0.01]
            }
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(test_yaml_dict, f)

        print(f"No YAML supplied, generated demo trajectory & config at {yaml_path}\n")
        test_yaml = str(yaml_path)

    main(test_yaml) 