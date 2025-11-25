import numpy as np
import yaml
import argparse
import scipy.interpolate as si
from scipy.interpolate import interp1d

from traj_util import slerpn, right_zero_hold, vis_traj

def random_xyz_bspline(dt, T, T_sample, start, max_speed, pos_max, pos_min):
    R = T_sample * max_speed
    N = int(T / T_sample)

    last_p = start.reshape(1, -1)
    key_points = [np.copy(last_p)] * 3
    for i in range(N):
        box_max = last_p + np.array([R, R, R])
        box_min = last_p - np.array([R, R, R])
        sample_max = np.minimum(box_max, pos_max)
        sample_min = np.maximum(box_min, pos_min)
        sample = np.random.uniform(sample_min, sample_max)
        key_points.append(sample)
        last_p = np.copy(sample)

    key_points.extend([np.copy(last_p)] * 3)
    key_points = np.concatenate(key_points)

    steps = int(T / dt)
    pos = bspline(key_points, n=steps, degree=5, periodic=False)

    return traj_from_pos(pos, dt)


def random_xyz_step(dt, T, T_sample, start, max_speed, pos_max, pos_min):
    R = T_sample * max_speed
    N = int(T / T_sample)

    last_p = start.reshape(1, -1)
    key_points = [np.copy(last_p)] * 1
    for i in range(N):
        box_max = last_p + np.array([R, R, R])
        box_min = last_p - np.array([R, R, R])
        sample_max = np.minimum(box_max, pos_max)
        sample_min = np.maximum(box_min, pos_min)
        sample = np.random.uniform(sample_min, sample_max)
        key_points.append(sample)
        last_p = np.copy(sample)
    
    # for i in range(N -1):
    #     print("p diff: ", key_points[i+1] - key_points[i], "R: ", R)

    key_points.extend([np.copy(last_p)] * 1)
    key_points = np.concatenate(key_points)
    T += 1 * T_sample
    t = np.linspace(0, T, len(key_points))
    # print(t[1] - t[0], T_sample, N)

    t_interp = np.linspace(0, T, int(T / dt))
    pos = np.zeros((len(t_interp), 3))
    for i in range(3):
        pos[:, i] = interp1d(t, key_points[:, i], kind='previous')(t_interp)
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    return pos, vel, acc



def xyz_figure8(dt, T, A, W, Phi, B):
    W = W.reshape(-1, 1)
    t = np.arange(0, T, dt).reshape(1, -1)
    Phi = Phi.reshape(-1, 1)
    A = A.reshape(-1, 1)
    B = B.reshape(-1, 1)
    pos = A * np.sin(W * t + Phi) + B
    return traj_from_pos(pos.T, dt)



def bspline(cv, n=100, degree=3, periodic=False):
    """Calculate n samples on a bspline

    cv :      Array ov control vertices
    n  :      Number of samples to return
    degree:   Curve degree
    periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count + degree + 1)
        factor, fraction = divmod(count + degree + 1, count)
        cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count - 1)
        kv = np.clip(np.arange(count + degree + 1) - degree, 0, count - degree)

    # Return samples
    max_param = count - (degree * (1 - periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))



def traj_from_pos(pos, dt):
    vel = np.diff(pos, axis=0) / dt
    vel = np.concatenate([vel, vel[-1:]], axis=0)
    acc = np.diff(vel, axis=0) / dt
    acc = np.concatenate([acc, acc[-1:]], axis=0)
    return pos, vel, acc




def save_traj(file_name, traj, keys):
    import os
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    np.savetxt(file_name, traj, delimiter=",", header=keys, comments="", fmt="%.6f")


def traj_gen(config):
    traj_type = config["traj_type"]
    traj_config = config[traj_type]
    dt = 0.01
    if traj_type == "random_xyz_bspline":
        T = traj_config["T"]
        T_sample = traj_config["T_sample"]
        start = np.array(traj_config["start"])
        max_speed = np.array(traj_config["max_speed"])
        pos_max = np.array(traj_config["pos_max"])
        pos_min = np.array(traj_config["pos_min"])
        traj = random_xyz_bspline(dt, T, T_sample, start, max_speed, pos_max, pos_min)
    
    elif traj_type == "figure8":
        T = traj_config["T"]
        A = np.array(traj_config["A"])
        W = np.array(traj_config["W"])
        Phi = np.array(traj_config["Phi"])
        B = np.array(traj_config["B"])
        traj = xyz_figure8(dt, T, A, W, Phi, B)
    
    elif traj_type == "step":
        T = traj_config["T"]
        T_sample = traj_config["T_sample"]
        start = np.array(traj_config["start"])
        max_speed = np.array(traj_config["max_speed"])
        pos_max = np.array(traj_config["pos_max"])
        pos_min = np.array(traj_config["pos_min"])
        traj = random_xyz_step(dt, T, T_sample, start, max_speed, pos_max, pos_min)
    
    else:
        raise ValueError("Unknown traj_type: %s" % traj_type)
    
    pos, vel, acc = traj
    
    traj = complete_traj(dt, pos, vel, acc)
    return traj

def complete_traj(dt, pos, vel, acc, quat=None, joint=None, gripper=None):

    traj_len = pos.shape[0]
    traj = np.zeros((traj_len, 16))
    
    t = np.arange(0, traj_len * dt, dt)
    
    if quat is None:
        quat = np.array([0, 0, 0, 1])
    if joint is None:
        joint = np.zeros((traj_len, 4))
    if gripper is None:
        gripper = np.zeros((traj_len,))

    P_IDX = 0
    QUAT_IDX = 3
    V_IDX = 7
    U_IDX = 10
    JOINT_IDX = 13
    GRIPPER_IDX = 17
    traj = np.zeros((len(t), 18))
    # interpolate position
    traj[:, P_IDX:P_IDX+3] = pos
    
    # interpolate velocity
    traj[:, V_IDX:V_IDX+3] = vel
    
    # interpolate acceleration
    traj[:, U_IDX:U_IDX+3] = acc
        
    # interpolate quaternion
    traj[:, QUAT_IDX:QUAT_IDX+4] = quat
    
    # # arm interp
    traj[:, JOINT_IDX:JOINT_IDX+4] = joint
    
    # # gripper interp
    traj[:, GRIPPER_IDX] = gripper
        
    traj_with_time = np.hstack((t.reshape(-1, 1), traj))
    
    return traj_with_time



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config", type=str, help="config file")
    argparser.add_argument("output", type=str, help="output file")
    args = argparser.parse_args()
    
    assert args.config is not None, "config file is not provided"
    assert args.output is not None, "output file is not provided"
    assert args.config.endswith(".yaml"), "config file should be yaml file"
    assert args.output.endswith(".csv"), "output file should be csv file"

    print("load config file: ", args.config)
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    output_file = args.output
    traj = traj_gen(config)
    
    print("Trajectory shape: ", traj.shape)
    
    keys = "t,px,py,pz,qx,qy,qz,qw,vx,vy,vz,ux,uy,uz,joint1,joint2,joint3,joint4,gripper"
    save_traj(output_file, traj, keys)
    vis_traj(traj, keys, output_file.replace(".csv", ".png"))
    
    print("Trajectory saved to ", output_file)
    print("Visualization saved to ", output_file.replace(".csv", ".png"))


if __name__ == "__main__":
    main()