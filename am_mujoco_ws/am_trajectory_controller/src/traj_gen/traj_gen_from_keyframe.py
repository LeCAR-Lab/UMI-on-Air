import numpy as np
import argparse

from traj_util import slerpn, right_zero_hold, vis_traj

def get_keyframe(file_name):
    keyframe = np.loadtxt(file_name, delimiter=",", skiprows=1)
    print("load keyframe, shape: ", keyframe.shape)
    
    return keyframe


def interp_keyframe_1d(keyframe, dt):
    assert keyframe.shape[1] == 12, "keyframe should have 12 columns, t, p, q, joint, gripper, but got %d" % keyframe.shape[1]


    time = keyframe[:, 0]
    pos, quat, joint, gripper = keyframe[:, 1:4], keyframe[:, 4:8], keyframe[:, 8:11], keyframe[:, 11]
    # normalize quaternion
    quat_norm = np.linalg.norm(quat, axis=1)
    quat = quat / quat_norm[:, None]

    assert abs(time[0] - 0) < 1e-6, "first time should be 0"

    t = np.arange(time[0], time[-1], dt)
    
    # traj = [t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,gripper]
    P_IDX = 0
    QUAT_IDX = 3
    V_IDX = 7
    U_IDX = 10
    JOINT_IDX = 13
    GRIPPER_IDX = 16
    traj = np.zeros((len(t), 17))
    # interpolate position
    for i in range(3):
        traj[:, P_IDX + i] = np.interp(t, time, pos[:, i])
    
    # interpolate velocity
    for i in range(3):
        traj[:, V_IDX + i] = np.gradient(traj[:, P_IDX + i], t)
        
    # interpolate quaternion
    traj[:, QUAT_IDX:QUAT_IDX+4] = slerpn(t, time, quat)
    
    # # arm interp
    for i in range(3):
        traj[:, JOINT_IDX + i] = np.interp(t, time, joint[:, i])
    
    # # gripper interp
    traj[:, GRIPPER_IDX] = right_zero_hold(t, time, gripper)
        
    traj_with_time = np.hstack((t.reshape(-1, 1), traj))
    
    return traj_with_time


def save_traj(file_name, traj, keys):
    np.savetxt(file_name, traj, delimiter=",", header=keys, comments="", fmt="%.6f")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("keyframe", type=str, help="key_frame.csv")
    argparser.add_argument("output", type=str, help="output file")
    args = argparser.parse_args()
    
    assert args.keyframe is not None, "keyframe file is not provided"
    assert args.output is not None, "output file is not provided"
    assert args.keyframe.endswith(".csv"), "keyframe file should be csv file"
    assert args.output.endswith(".csv"), "output file should be csv file"

    dt = 0.01
    keyframe_file = args.keyframe
    output_file = args.output
    key_frame = get_keyframe(keyframe_file)
    traj= interp_keyframe_1d(key_frame, dt)
    print("Trajectory shape: ", traj.shape)
    
    keys = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,gripper"
    header_key = "t,px,py,pz,qx,qy,qz,qw,joint1,joint2,gripper"
    save_traj(output_file, traj, keys)
    vis_traj(traj, keys, output_file.replace(".csv", ".png"), key_frame, header_key)
    
    print("Trajectory saved to ", output_file)
    print("Visualization saved to ", output_file.replace(".csv", ".png"))


if __name__ == "__main__":
    main()
