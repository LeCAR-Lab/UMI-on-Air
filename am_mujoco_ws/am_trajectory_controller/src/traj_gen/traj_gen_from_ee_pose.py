import numpy as np
import argparse

from traj_util import slerpn

def get_keyframe(file_name):
    keyframe = np.loadtxt(file_name, delimiter=",", skiprows=1)
    print("load keyframe, shape: ", keyframe.shape)
    
    return keyframe


def interp_keyframe_1d(keyframe, dt):
    assert keyframe.shape[1] == 8, "keyframe should have 8 columns, t, p, q, but got %d" % keyframe.shape[1]

    time = keyframe[:, 0]
    ee_pos = keyframe[:, 1:4]
    ee_quat = keyframe[:, 4:8]
    # normalize quaternion
    ee_quat_norm = np.linalg.norm(ee_quat, axis=1)
    ee_quat = ee_quat / ee_quat_norm[:, None]
    assert abs(time[0] - 0) < 1e-6, "first time should be 0"

    t = np.arange(time[0], time[-1], dt)

    # traj = [t,px,py,pz,qx,qy,qz,qw]
    P_IDX = 0
    QUAT_IDX = 3
    traj = np.zeros((len(t), 7))
    # interpolate position
    for i in range(3):
        traj[:, P_IDX + i] = np.interp(t, time, ee_pos[:, i])
    # interpolate quaternion
    traj[:, QUAT_IDX:QUAT_IDX+4] = slerpn(t, time, ee_quat)
    
    traj_with_time = np.hstack((t.reshape(-1, 1), traj))
    
    return traj_with_time


def save_traj(file_name, traj):
    np.savetxt(file_name, traj, delimiter=",", header="t,px,py,pz,qx,qy,qz,qw", comments="", fmt="%.6f")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("keyframe", type=str, help="key_frame.csv")
    argparser.add_argument("output", type=str, help="output file")
    args = argparser.parse_args()
    
    assert args.keyframe is not None, "keyframe file is not provided"
    assert args.output is not None, "output file is not provided"

    dt = 0.01
    keyframe_file = args.keyframe
    output_file = args.output
    key_frame = get_keyframe(keyframe_file)
    traj= interp_keyframe_1d(key_frame, dt)
    print("Trajectory shape: ", traj.shape)

    keys = "t,px,py,pz,qx,qy,qz,qw"
    header_key = "t,px,py,pz,qx,qy,qz,qw"

    save_traj(output_file, traj)
    # vis_traj(traj, keys, output_file.replace(".csv", ".png"), key_frame, header_key)
    print("Trajectory saved to ", output_file)


if __name__ == "__main__":
    main()
