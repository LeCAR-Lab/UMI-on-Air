from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import numpy as np
import quaternion
import mujoco
import math
from spatialmath import base
import matplotlib.pyplot as plt

def calculate_arm_Te(pose, quate):
    """
    Calculate the pose transform matrix of the end-effector.
    """
    if type(quate) is quaternion.quaternion:
        arm_ee_quat = quate
    else:
        arm_ee_quat = np.quaternion(quate[0], quate[1], quate[2], quate[3])
    # Calculate forward kinematics (Tep) for the target end-effector pose
    res = np.zeros(9)
    mujoco.mju_quat2Mat(res, np.array([arm_ee_quat.w, arm_ee_quat.x, arm_ee_quat.y, arm_ee_quat.z]))
    Te = np.eye(4)
    Te[:3,3] = pose
    Te[:3,:3] = res.reshape((3,3))
    return Te

def angle_axis_python(T, Td):
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if base.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln
    e[3:] = a

    return e

def slerpn(t, tp, qp):
    slerp = Slerp(tp, R.from_quat(qp))
    interpolated_rots = slerp(t)
    return interpolated_rots.as_quat()

def right_zero_hold(x, xp, fp):
    f = interp1d(xp, fp, kind='previous')
    return f(x)


def vis_traj(traj, header, path, keyframe=None, key_header=None):
    header = header.split(",")
    N = len(header) - 1
    rows = 4
    cols = N // rows if N % rows == 0 else N // rows + 1
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, 20))
    
    t = traj[:, 0]
    
    if keyframe is not None:
        assert key_header is not None, "key_header should be provided"
        
        t_key = keyframe[:, 0]
        header_key = key_header.split(",")
    # print(N, rows, cols)
    for i in range(N):
        c = i // rows
        r = i % rows
        axs[r, c].plot(t, traj[:, i + 1])
        axs[r, c].set_title(header[i + 1])
        axs[r, c].set_xlabel("time")
        if keyframe is not None and header[i + 1] in header_key:
            idx = header_key.index(header[i + 1])
            axs[r, c].scatter(t_key, keyframe[:, idx], c="r")
        axs[r, c].grid()
        y_range = np.max(traj[:, i + 1]) - np.min(traj[:, i + 1])
        if y_range < 0.01:
            axs[r, c].set_ylim([np.min(traj[:, i + 1]) - 0.01, np.max(traj[:, i + 1]) + 0.01])
        
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)