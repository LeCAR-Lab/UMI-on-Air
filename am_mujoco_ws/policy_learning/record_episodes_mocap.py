import os
import time
import h5py
import argparse
import numpy as np
import cv2
import tkinter as tk

from constants import SIM_TASK_CONFIGS, DT
from ee_sim_env import make_ee_sim_env
from mocap_connection import initialize_mocap_connection, get_mocap_data, reset_calibration
from episode_utils import get_auto_index

# DM Control viewer for 3D visualization
from dm_control import viewer
VIEWER_AVAILABLE = True

def main(args):
    # Initialize mocap connection
    connection_event = initialize_mocap_connection()
    print("⌛ Waiting for mocap connection...")
    connection_event.wait()
    print("✅ Mocap connection established. Proceeding with simulation.")
    
    task_name = args['task_name']
    onscreen_render = args['onscreen_render']
    show_3d_viewer = args.get('show_3d_viewer', False)
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    render_cam_name = 'ee'

    if args['episode_idx'] is not None:
        episode_idx = args['episode_idx']
    else:
        episode_idx = get_auto_index(dataset_dir)

    env = make_ee_sim_env(task_name, camera_names=camera_names)
    # Reset mocap calibration after env is created so first mocap frame anchors
    # to the task-specific reference_ee_pos set by the env's task.
    reset_calibration()

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    use_direct_motion = True
    policy = None  # Will use direct motion instead

    # Launch 3D viewer if requested
    if show_3d_viewer and VIEWER_AVAILABLE:
        print("Launching 3D viewer...")
        print("Controls:")
        print("  SPACE - Start/pause simulation")
        print("  ESC - Exit viewer")
        print("  Mouse - Rotate camera")
        print("  Mouse wheel - Zoom")
        print("  Ctrl+Mouse - Pan camera")
        
        # Create a simple policy function for the viewer
        step_counter = [0]  # Use list to make it mutable in closure
        
        def viewer_policy(timestep):
            action = get_mocap_data()
            step_counter[0] += 1
            # Reset counter if we've reached the end
            if step_counter[0] >= episode_len:
                step_counter[0] = 0
            return action
        
        # Launch the interactive viewer
        viewer.launch(env, policy=viewer_policy)
        return  # Exit after viewer closes

    success = []

    steps_until_stop_success = int(3.0 / DT)

    # ---------------------------------------------------------
    # Helper to save a **successful** episode to disk.  Episodes
    # that reach this point are guaranteed to have length
    # len(action_traj) == len(episode) - 1.
    # ---------------------------------------------------------

    def _save_episode(dataset_dir, episode_idx, episode, action_traj):
        """Write episode to <dataset_dir>/episode_<idx>.hdf5 atomically."""
        if len(action_traj) == 0:
            print("[WARN] No timesteps – nothing to save.")
            return

        max_timesteps = len(action_traj)

        # ------------------------------------------------------------------
        # Build arrays first so we know their exact shapes – no assumptions!
        # ------------------------------------------------------------------

        # Augment qpos with the gripper command so it matches the 8-element
        # joint layout used in the keyboard recordings (EE pose + gripper).
        qpos_arr = np.stack([
            np.concatenate([episode[i].observation['qpos'], [action_traj[i][7]]])
            for i in range(max_timesteps)
        ])
        act_arr = np.stack(action_traj)
        # Ensure gripper command is within [0,1]
        act_arr[:, 7] = np.clip(act_arr[:, 7], 0.0, 1.0)  # (T,8)

        # Dummy qvel – mocap pipeline does not provide it, but we keep the
        # dataset to stay schema-compatible with keyboard recordings.
        qvel_arr = np.zeros((max_timesteps, 6), dtype=np.float64)

        img_arrays = {}
        for cam_name in camera_names:
            img_arrays[cam_name] = np.stack([episode[i].observation['images'][cam_name] for i in range(max_timesteps)])

        # ------------------------------------------------------------------
        # Write HDF5 – create each dataset with the exact shape we have.
        # ------------------------------------------------------------------

        tmp_path = os.path.join(dataset_dir, f'episode_{episode_idx}.tmp')
        final_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')

        t0 = time.time()
        with h5py.File(tmp_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True

            obs_grp = root.create_group('observations')
            # simple datasets
            obs_grp.create_dataset('qpos', data=qpos_arr)
            obs_grp.create_dataset('qvel', data=qvel_arr)
            root.create_dataset('action', data=act_arr)

            # image datasets
            img_grp = obs_grp.create_group('images')
            for cam_name, arr in img_arrays.items():
                img_grp.create_dataset(cam_name, data=arr, dtype='uint8', chunks=(1,) + arr.shape[1:])

            # Mark episode as successful (mocap saver only writes successes)
            root.attrs['success'] = True

        os.replace(tmp_path, final_path)
        print(f"💾 Saved to {final_path} ({time.time() - t0:.1f}s)")

    # OpenCV window setup for display (created once)
    window_name = 'Simulation Preview'
    if onscreen_render:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Make window fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Get screen dimensions for proper scaling
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        
        print(f"Screen resolution: {screen_width}x{screen_height}")
        
        # Display initial frame
        initial_img = env.reset().observation['images'][render_cam_name]
        # Convert RGB to BGR for OpenCV display
        initial_img_bgr = cv2.cvtColor(initial_img, cv2.COLOR_RGB2BGR)
        # Resize to fill screen
        initial_img_resized = cv2.resize(initial_img_bgr, (screen_width, screen_height))
        cv2.imshow(window_name, initial_img_resized)

    while True:  # loop over episodes
        print(f"\n==============================\nRecording episode_{episode_idx}.hdf5\n==============================")
        exit_program = False
        restart_episode = False
        ts = env.reset()
        episode = [ts]
        action_traj = []

        # Display initial frame for ready screen
        if onscreen_render:
            initial_img = ts.observation['images'][render_cam_name]
            initial_img_bgr = cv2.cvtColor(initial_img, cv2.COLOR_RGB2BGR)
            initial_img_resized = cv2.resize(initial_img_bgr, (screen_width, screen_height))
            cv2.imshow(window_name, initial_img_resized)

        print("🎬 Ready to record!")
        print("SPACE – start | ESC – exit | F – fullscreen | X – delete last")

        recording_started = False
        while not recording_started:
            if onscreen_render:
                # Get current state for display
                if use_direct_motion:
                    action = get_mocap_data()
                else:
                    action = policy.get_action()
                
                ts_current = env.step(action)
                current_img = ts_current.observation['images'][render_cam_name]
                current_img_bgr = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
                # Resize to fill screen
                current_img_resized = cv2.resize(current_img_bgr, (screen_width, screen_height))
                
                # Add text overlay (scaled for larger screen)
                epi_text = f"EPISODE {episode_idx}"
                text1 = "READY TO RECORD"
                text2 = "Press SPACE to start recording"
                text3 = "ESC-exit | F-fullscreen | X-delete last"
                
                cv2.putText(current_img_resized, epi_text, (50, screen_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 5)
                cv2.putText(current_img_resized, text1, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 6)
                cv2.putText(current_img_resized, text2, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                cv2.putText(current_img_resized, text3, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 3)
                
                cv2.imshow(window_name, current_img_resized)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):  # Space bar – initiate 3-sec countdown
                countdown_secs = 3
                print("🚦 Countdown to recording…")
                for sec in range(countdown_secs, 0, -1):
                    if onscreen_render:
                        cnt_img = current_img_resized.copy()
                        cv2.putText(cnt_img, str(sec), (screen_width // 2 - 50, screen_height // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 12)
                        cv2.imshow(window_name, cnt_img)
                        cv2.waitKey(1000)  # wait 1 sec per number
                    else:
                        time.sleep(1)
                recording_started = True
                print("🔴 Started recording")
                episode = [ts_current]
                break
            elif key == 27:  # ESC key to exit program
                print("Exit requested")
                exit_program = True
                break
            elif key == ord('x') or key == ord('X'):
                # Delete the most recently saved episode, if any
                if episode_idx == 0:
                    print("🚫 No previous episode to delete.")
                else:
                    last_idx = episode_idx - 1
                    last_path = os.path.join(dataset_dir, f'episode_{last_idx}.hdf5')
                    if os.path.isfile(last_path):
                        os.remove(last_path)
                        episode_idx -= 1
                        print(f"🗑️  Deleted {last_path}. Now recording episode_{episode_idx}.")
                    else:
                        print("⚠️  Expected file not found:" , last_path)
                continue  # refresh READY screen with updated index
            elif key == ord('f') or key == ord('F'):  # F key to toggle fullscreen
                # Toggle fullscreen
                current_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
                if current_prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("Switched to windowed mode")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Switched to fullscreen mode")
        
        if exit_program:
            break  # break outer while loop

        step = 0
        success_detected = False
        success_counter = 0

        while True:
            if use_direct_motion:
                # Generate motion directly for UMI task
                action = get_mocap_data()
            else:
                # Use policy for other tasks
                action = policy.get_action()
            
            action_traj.append(action)
            ts = env.step(action)
            episode.append(ts)
            
            if onscreen_render:
                # Convert RGB to BGR for OpenCV display
                img_bgr = cv2.cvtColor(ts.observation['images'][render_cam_name], cv2.COLOR_RGB2BGR)
                # Resize to fill screen
                img_resized = cv2.resize(img_bgr, (screen_width, screen_height))
                
                # Add recording status overlay (scaled for larger screen)
                overlay_text = f"EP{episode_idx} REC {step+1}/{episode_len} | R-abort | ESC-exit"
                if success_detected:
                    remaining = max(0, (steps_until_stop_success - success_counter) * DT)
                    overlay_text = f"SUCCESS – stopping in {remaining:.1f}s"
                cv2.putText(img_resized, overlay_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                
                cv2.imshow(window_name, img_resized)
                # Check for ESC key to stop recording early
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to exit program
                    print("Exit requested")
                    exit_program = True
                    break
                elif key in (ord('r'), ord('R')):  # R to abort and restart episode
                    print("🔁 Restart requested – discarding current episode.")
                    restart_episode = True
                    break
            
            step += 1
            
            # Success countdown handling
            if not success_detected and ts.reward == env.task.max_reward:
                success_detected = True
                success_counter = 0
                print("✔️  Success reached – finishing in 3s")

            if success_detected:
                success_counter += 1
                if success_counter >= steps_until_stop_success:
                    print("Success countdown elapsed – stopping recording")
                    break

            if step >= episode_len:
                break

            if not use_direct_motion and not policy.is_recording():
                break
        
        if exit_program:
            break  # break outer while loop

        # -------------------------------------------------
        # Decide what to do with the just-recorded episode
        # -------------------------------------------------

        success_episode = success_detected and success_counter >= steps_until_stop_success

        if restart_episode or not success_episode:
            if restart_episode:
                print("🔄 Episode restarted – ready for new attempt.")
            else:
                print("⚠️  Episode did not reach success – discarding.")
            continue  # start outer loop again without saving

        # Episode successful → save to disk and advance index
        _save_episode(dataset_dir, episode_idx, episode[:-1], action_traj)
        episode_idx += 1
        continue  # start next episode

    if onscreen_render:
        cv2.destroyAllWindows()

    return  # graceful exit once user pressed ESC


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--show_3d_viewer', action='store_true', help='Show 3D MuJoCo viewer window')

    main(vars(parser.parse_args()))