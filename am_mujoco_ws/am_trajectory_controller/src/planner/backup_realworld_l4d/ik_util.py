import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R


def ee_pos_to_base_pos_and_arm_joints(ik, target_ee_pos, target_ee_ori, cur_base_pos, cur_base_ori, cur_joints, drone_state):
    '''
    input: 
    target_ee_ori: x,y,z,w
    cur_base_ori: x,y,z,w
    '''    
    # print("cur_joints", cur_joints)
    # print("ee_pos_target", ee_pos_target)
    current_qpos = np.zeros(11)
    current_qpos[:3] = cur_base_pos
    # gazebo quat: x,y,z,w -> mujoco quat: w,x,y,z
    current_qpos[3:7] = [cur_base_ori[3], cur_base_ori[0], cur_base_ori[1], cur_base_ori[2]]
    current_qpos[7:9] = cur_joints
    # gazebo quat: x,y,z,w -> mujoco quat: w,x,y,z
    target_ee_quat = np.array([target_ee_ori[3], target_ee_ori[0], target_ee_ori[1], target_ee_ori[2]])
    try:
        km_target = ik.optimize(current_qpos, target_ee_pos, target_ee_quat, drone_state)
    except:
        km_target = None

    return km_target


def ee_pos_to_base_pos_and_arm_joints_4dof(ik, target_ee_pos, target_ee_ori, cur_base_pos, cur_base_ori, cur_joints, drone_state):
    '''
    input: 
    target_ee_ori: x,y,z,w
    cur_base_ori: x,y,z,w

    output:
    km_target

    '''    
    # print("cur_joints", cur_joints)
    # print("ee_pos_target", ee_pos_target)
    current_qpos = np.zeros(12)
    current_qpos[:3] = cur_base_pos
    # gazebo quat: x,y,z,w -> mujoco quat: w,x,y,z
    current_qpos[3:7] = [cur_base_ori[3], cur_base_ori[0], cur_base_ori[1], cur_base_ori[2]]
    current_qpos[7:10] = cur_joints
    # gazebo quat: x,y,z,w -> mujoco quat: w,x,y,z
    target_ee_quat = np.array([target_ee_ori[3], target_ee_ori[0], target_ee_ori[1], target_ee_ori[2]])
    try:
        km_target = ik.optimize(current_qpos, target_ee_pos, target_ee_quat, drone_state)
    except:
        km_target = None

    return km_target



############################################################################################################
# CasADi rotation matrix and quaternion conversion functions


def rotation_matrix_from_quaternion(quaternion):
    qw, qx, qy, qz = quaternion
    quaternion = np.array([qx, qy, qz, qw])
    r = R.from_quat(quaternion)
    return r.as_matrix()

def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = quaternion
    quaternion = np.array([qx, qy, qz, qw])
    r = R.from_quat(quaternion)
    return r.as_matrix()

def quaternion_from_rotation_matrix(rotation_matrix):
    # print("rotation_matrix: ", rotation_matrix)
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    qx, qy, qz, qw = quaternion
    quaternion = np.array([qw, qx, qy, qz])
    return quaternion

def rotation_matrix_to_quaternion(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    quaternion = r.as_quat()
    qx, qy, qz, qw = quaternion
    quaternion = np.array([qw, qx, qy, qz])
    return quaternion

def quaternion_to_rpy(quaternion):
    qw, qx, qy, qz = quaternion
    quaternion = np.array([qx, qy, qz, qw])
    r = R.from_quat(quaternion)
    euler = r.as_euler('zyx', degrees=False)
    euler = euler[::-1]
    return euler

def rpy_to_rotation_matrix(rpy):
    ypr = rpy[::-1]
    r = R.from_euler('zyx', ypr, degrees=False)
    return r.as_matrix()

def euler_to_rotation_matrix(euler):
    ypr = euler[::-1]
    r = R.from_euler('zyx', ypr, degrees=False)
    return r.as_matrix()


def rotation_matrix_from_quaternion_ca(q):
    # Ensure q is a casadi.MX type with 4 elements [q_w, q_x, q_y, q_z]
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]

    # Squares of the components
    q_w2 = q_w*q_w
    q_x2 = q_x*q_x
    q_y2 = q_y*q_y
    q_z2 = q_z*q_z

    # Cross products
    qxqy = q_x * q_y
    qxqz = q_x * q_z
    qyqz = q_y * q_z
    qxqw = q_x * q_w
    qyqw = q_y * q_w
    qzqw = q_z * q_w

    # Rotation matrix elements
    r11 = 1 - 2 * (q_y2 + q_z2)
    r12 = 2 * (qxqy - qzqw)
    r13 = 2 * (qxqz + qyqw)
    r21 = 2 * (qxqy + qzqw)
    r22 = 1 - 2 * (q_x2 + q_z2)
    r23 = 2 * (qyqz - qxqw)
    r31 = 2 * (qxqz - qyqw)
    r32 = 2 * (qyqz + qxqw)
    r33 = 1 - 2 * (q_x2 + q_y2)

    # Constructing the rotation matrix
    R = ca.MX(3, 3)  # Create a 3x3 matrix
    R[0, 0] = r11
    R[0, 1] = r12
    R[0, 2] = r13
    R[1, 0] = r21
    R[1, 1] = r22
    R[1, 2] = r23
    R[2, 0] = r31
    R[2, 1] = r32
    R[2, 2] = r33

    return R

def quaternion_from_rotation_matrix_ca(R):
    """
    Convert a rotation matrix to a quaternion using CasADi symbolic operations.

    Parameters:
    R (casadi.SX or casadi.MX): A 3x3 rotation matrix.

    Returns:
    q (casadi.SX or casadi.MX): A 4-element quaternion vector [qw, qx, qy, qz].
    """
    # Compute the elements of the quaternion
    qw = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 + R[0, 0] + R[1, 1] + R[2, 2]))
    qx = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 + R[0, 0] - R[1, 1] - R[2, 2]))
    qy = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 - R[0, 0] + R[1, 1] - R[2, 2]))
    qz = 0.5 * ca.sqrt(ca.fmax(1e-7, 1 - R[0, 0] - R[1, 1] + R[2, 2]))

    # Adjust the signs of the quaternion components
    qx = ca.if_else((R[2, 1] - R[1, 2]) < 0, -qx, qx)
    qy = ca.if_else((R[0, 2] - R[2, 0]) < 0, -qy, qy)
    qz = ca.if_else((R[1, 0] - R[0, 1]) < 0, -qz, qz)

    # Assemble the quaternion vector
    q = ca.vertcat(qw, qx, qy, qz)

    return q


def rotation_matrix_from_euler_ca(euler):
    """
    Convert Euler angles to a rotation matrix using CasADi symbolic operations in yaw-pitch-roll order.

    Parameters:
    euler (casadi.SX or casadi.MX): A 3-element vector of Euler angles [yaw, pitch, roll].

    Returns:
    R (casadi.SX or casadi.MX): A 3x3 rotation matrix.

    Note:
    The rotations are applied in the yaw-pitch-roll order.
    """
    # Extract the Euler angles
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    
    # Compute the sine and cosine of the Euler angles
    c1 = ca.cos(yaw)    # cos(yaw)
    c2 = ca.cos(pitch)  # cos(pitch)
    c3 = ca.cos(roll)   # cos(roll)
    s1 = ca.sin(yaw)    # sin(yaw)
    s2 = ca.sin(pitch)  # sin(pitch)
    s3 = ca.sin(roll)   # sin(roll)

    # Construct the rotation matrix
    R = ca.MX.zeros(3, 3)  # Create a 3x3 matrix

    # First row
    R[0, 0] = c1 * c2
    R[0, 1] = c1 * s2 * s3 - s1 * c3
    R[0, 2] = c1 * s2 * c3 + s1 * s3

    # Second row
    R[1, 0] = s1 * c2
    R[1, 1] = s1 * s2 * s3 + c1 * c3
    R[1, 2] = s1 * s2 * c3 - c1 * s3

    # Third row
    R[2, 0] = -s2
    R[2, 1] = c2 * s3
    R[2, 2] = c2 * c3

    return R
