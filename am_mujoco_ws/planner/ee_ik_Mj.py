import mujoco
import numpy as np
import quaternion
from abc import ABC, abstractmethod
from typing import Optional
import qpsolvers as qp
from functools import wraps
import warnings
from spatialmath import base
import math
from spatialmath import base
from scipy.spatial.transform import Rotation as R
import time
import mujoco.viewer
warnings.filterwarnings('ignore')


def init_ik_Mj(xml_path, cur_base_pos, cur_base_quat, cur_joints, visualize):
    ik = MjIKPlanner(xml_path, visualize=visualize)
    ik.data.qpos = np.zeros(11)
    ik.data.qpos[:3] = cur_base_pos
    # gazebo quat: x,y,z,w -> mujoco quat: w,x,y,z
    ik.data.qpos[3:7] = [cur_base_quat[3], cur_base_quat[0], cur_base_quat[1], cur_base_quat[2]]
    ik.data.qpos[7:9] = cur_joints
    mujoco.mj_fwdPosition(ik.model, ik.data)
    print("mujoco_ee: ", ik.data.body("ee").xpos)
    return ik



def rotate_quaternion_yaw(q, delta_yaw):
    r = R.from_quat(q)
    yaw_rotation = R.from_euler('z', delta_yaw)
    r_new = r * yaw_rotation
    return r_new.as_quat()

def calculate_arm_Te(pose, quate):
    """
    Calculate the pose transform matrix of the end-effector.
    """
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

class IK(ABC):
    """
    An abstract super class which provides basic functionality to perform numerical inverse
    kinematics (IK). Superclasses can inherit this class and implement the solve method.
    """

    def __init__(
        self,
        name: str = "IK Solver",
        ilimit: int = 30,
        tol: float = 1e-6,
        we: np.ndarray = np.ones(6),
        problems: int = 1000,
        reject_jl: bool = True,
        λΣ: float=0.0,
        λm: float=0.0, 
        ps: float=0.1,
        pi: Optional[np.ndarray]=None,
    ):
        """
        name: The name of the IK algorithm
        ilimit: How many iterations are allowed within a search before a new search is started
        tol: Maximum allowed residual error E
        we: A 6 vector which assigns weights to Cartesian degrees-of-freedom
        reject_jl: Reject solutions with joint limit violations
        λΣ: The gain for joint limit avoidance. Setting to 0.0 will remove this completely from the solution
        λm: The gain for maximisation. Setting to 0.0 will remove this completely from the solution
        ps: The minimum angle/distance (in radians or metres) in which the joint is allowed to approach to its limit
        """

        # Solver parameters
        self.name = name
        self.ilimit = ilimit
        self.tol = tol
        self.we = we
        self.We = np.diag(we)
        self.reject_jl = reject_jl
        self.λΣ = λΣ
        self.λm = λm
        self.ps = ps

    def solve(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray):
        """
        This method will attempt to solve the IK problem and obtain joint coordinates
        which result the the end-effector pose Tep.

        The method returns a tuple:
        q: The joint coordinates of the solution (ndarray). Note that these will not
            be valid if failed to find a solution
        success: True if a solution was found (boolean)
        iterations: The number of iterations it took to find the solution (int)
        searches: The number of searches it took to find the solution (int)
        residual: The residual error of the solution (float)
        jl_valid: True if joint coordinates q are within the joint limits
        total_t: The total time spent within the step method
        """
        error = -1
        # Iteration count
        i = 0
        total_i = 0
        total_t = 0.0
        q = np.zeros(model.nv)
        q = data.qpos.copy()
        q_solved = np.zeros(q.shape)

        while i <= self.ilimit:
            i += 1
            # Attempt a step
            try:
                t, E, q = self.step(model, data, Tep, q, i)
                error = E
                q_solved[:] = q[:]
                # print("E: ", E)
                # Acclumulate total time
                total_t += t
            except np.linalg.LinAlgError:
                # Abandon search and try again
                print("break LinAlgError")
                break

            # Check if we have arrived
            if E < self.tol:
                # Wrap q to be within +- 180 deg
                # If your robot has larger than 180 deg range on a joint
                # this line should be modified in incorporate the extra range
                q_arm = q[self.actuator_idx+1].copy()
                q_arm = (q_arm + np.pi) % (2 * np.pi) - np.pi
                # Check if we have violated joint limits
                jl_valid = self.check_jl(model, q_arm)

                if not jl_valid and self.reject_jl:
                    # Abandon search and try again
                    print("q[self.actuator_idx+1]: ", q[self.actuator_idx+1])
                    break
                else:
                    # print("q_solved: {}, error: {}".format(q_solved, error))
                    # print("iteration: {}, total_t: {}".format(i, total_t))
                    # print("solved ik!! \n")
                    return q, True, total_i + i, E, jl_valid, total_t

        total_i += i
        i = 0

        # If we make it here, then we have failed
        return q, False, np.nan, np.nan, E, False, np.nan

    def error(self, Te: np.ndarray, Tep: np.ndarray, n_target: int):
        """
        Calculates the engle axis error between current end-effector pose Te and
        the desired end-effector pose Tep. Also calulates the quadratic error E
        which is weighted by the diagonal matrix We.

        Returns a tuple:
        e: angle-axis error (ndarray in R^6)
        E: The quadratic error weighted by We
        """
        # e = rtb.angle_axis(Te, Tep)
        assert n_target == 3 or n_target == 5 or n_target == 6, "3 for position only, 5 for position, pitch and yaw, 6 for position and orientation"
        e = angle_axis_python(Te, Tep)
        # E = 0.5 * e[0:n_target] @ self.We[0:n_target,0:n_target] @ e[0:n_target]
        e_pos = e[0:3]; we_pos = self.we[0:3]
        e_angle = e[4:]; we_angle = self.we[4:]
        e = np.concatenate((e_pos, e_angle), axis=0)
        We = np.diag(np.concatenate((we_pos, we_angle), axis=0))
        E = 0.5 * e @ We @ e

        return e, E

    def check_jl(self, model: mujoco.MjModel, q_arm: np.ndarray):
        """
        Checks if the joints are within their respective limits

        Returns a True if joints within feasible limits otherwise False
        """
        # Loop through the joints in the ETS
        for i in range(len(self.joint_idx)):
            # Get the corresponding joint limits
            ql0 = model.joint(self.joint_idx[i]).range[0]
            ql1 = model.joint(self.joint_idx[i]).range[1]
            # Check if q exceeds the limits
            if q_arm[i] < ql0 or q_arm[i] > ql1:
                return False

        # If we make it here, all the joints are fine
        return True

    @abstractmethod
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray, i: int):
        """
        Superclasses will implement this method to perform a step of the implemented
        IK algorithm
        """
        pass

def timing(func):
    @wraps(func)
    def wrap(*args, **kw):
        t_start = time.time()
        E, q = func(*args, **kw)
        t_end = time.time()
        t = t_end - t_start
        return t, E, q
    return wrap

class QP(IK):
    def __init__(self, name="QP", pz_l=0.1, pz_h=0.8, λj=1.0, λs=1.0, λxy=1.0, λz=1.0, λyaw=1.0,
                 joint_idx=np.array([1, 2]), actuator_idx=np.array([6, 7]), **kwargs):
        super().__init__(name, **kwargs)

        self.name = f"QP (λj={λj}, λs={λs})"
        self.pz_l = pz_l
        self.pz_h = pz_h
        self.λj = λj
        self.λs = λs
        self.λxy = λxy
        self.λz = λz
        self.λyaw = λyaw
        self.joint_idx = joint_idx
        self.actuator_idx = actuator_idx

        if self.λΣ > 0.0:
            self.name += ' Σ'

        if self.λm > 0.0:
            self.name += ' Jm'

    @timing
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, Tep: np.ndarray, q: np.ndarray, i: int):
        # Dimention of the action space [x, y, z, yaw, manipulator_link1_pitch, manipulator_link2_pitch]
        n_act = 6
        # Dimension of the base state: 4 for position and yaw
        n_to = 4
        # Dimension of the joint space: 2 for the pitch links
        n_joint = 2
        # Dimension of the ee state: 3 for position only, 6 for position and orientation
        n_target = 5
        # Calculate forward kinematics (Te)
        data.qpos[:] = q[:]
        mujoco.mj_fwdPosition(model, data)
        Te = calculate_arm_Te(data.body("ee").xpos, data.body("ee").xquat)
        # Calculate the error
        e, E = self.error(Te, Tep, 5)
        if E < self.tol and i <= 1:
            # print("NO NEED to calculate IK!!!!!!!!!!!!!")
            return E, q
        
        # Calculate the Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, model.body("ee").id)
        jac = np.concatenate((jacp, jacr[1:, :]), axis=0) # ee: x, y, z, pitch, yaw
        # Note: Customize the Jacobian
        J_base = np.concatenate((jac[:, :3], jac[:, 5:6]), axis=1) # base: x, y, z, yaw
        J_arm = jac[:, self.actuator_idx] # arm: pitch1, pitch2
        J = np.concatenate((J_base, J_arm), axis=1)
        # Quadratic component of objective function
        Q = np.eye(n_act + n_target)
        # Joint velocity component of Q
        Q[: n_act, : n_act] *= self.λj
        # Adjust the joint velocity component of Q
        Q[: 2, : 2] *= self.λxy
        Q[2, 2] *= self.λz
        Q[3, 3] *= self.λyaw
        # Slack component of Q
        Q[n_act :, n_act :] = self.λs * (1 / np.sum(np.abs(e))) * np.eye(5)
        # The equality contraints
        Aeq = np.concatenate((J, np.eye(n_target)), axis=1)
        beq = 2*e.reshape((n_target,))

        # The inequality constraints for joint limit avoidance
        if self.λΣ > 0.0:
            Ain = np.zeros((n_act, n_act + n_target))
            bin = np.zeros(n_act)

            # Form the joint limit velocity damper
            Ain_l = np.zeros((n_act, n_act))
            Bin_l = np.zeros(n_act)

            for i in range(n_joint):
                ql0 = model.joint(self.joint_idx[i]).range[0]
                ql1 = model.joint(self.joint_idx[i]).range[1]
                # Calculate the influence angle/distance (in radians or metres) in null space motion becomes active
                pi = (model.joint(self.joint_idx[i]).range[1] - model.joint(self.joint_idx[i]).range[0])/2

                if ql1 - q[self.actuator_idx[i]+1] <= pi:
                    Bin_l[i+n_to] = ((ql1 - q[self.actuator_idx[i]+1]) - self.ps) / (pi - self.ps)
                    Ain_l[i+n_to, i+n_to] = 1

                if q[self.actuator_idx[i]+1] - ql0 <= pi:
                    Bin_l[i+n_to] = -(((ql0 - q[self.actuator_idx[i]+1]) + self.ps) / (pi - self.ps))
                    Ain_l[i+n_to, i+n_to] = -1

            Ain[: n_act, : n_act] = Ain_l
            bin[: n_act] =  (1.0 / self.λΣ) * Bin_l
        else:
            Ain = None
            bin = None
        
        # Add self-collision avoidance in z-axis
        Ain = np.vstack([Ain, np.zeros((2, Ain.shape[1]))])
        Ib = np.zeros((n_target, n_act))
        Ib[:, :n_target] = np.identity(n_target)
        base_z = q[2]
        ee_z = data.body("ee").xpos[2]
        # Prevent the end-effector from going down into the base
        Ain[-2, :] = np.concatenate((Ib-J, np.zeros((n_target, n_target))), axis=1)[2, :]
        bin = np.append(bin, ee_z - base_z - self.pz_l)
        # Prevent the end-effector from going up too high relative to the base
        Ain[-1, :] = np.concatenate((J-Ib, np.zeros((n_target, n_target))), axis=1)[2, :]
        bin = np.append(bin, base_z - ee_z + self.pz_h)

        # Solve the QP
        c = np.zeros(n_act + n_target)
        xd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=None, ub=None, solver='quadprog')
        q[:3] += xd[:3]
        current_quat = [q[4], q[5], q[6], q[3]] # [x, y, z, w]
        next_quat = rotate_quaternion_yaw(current_quat, xd[3]) # [x, y, z, w]
        q[3:7] = [next_quat[3], next_quat[0], next_quat[1], next_quat[2]] # [w, x, y, z]
        q[self.actuator_idx+1] += xd[n_to:n_act]

        return E, q

class MjIKPlanner:
    """
    Inverse kinematics planner for the end-effector
    """
    def __init__(self, xml_path, ps=0.1, pz_l=0.0, pz_h=0.9,
                 λΣ=1e3, λj=2, λs=0.1, tol=1e-5, 
                 λxy=0.1, λz=0.04, λyaw=1, 
                 ilimit=100, visualize=False):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.visualize = visualize
        DEFAULT = 0; TAKEOFF = 1; PAUSE = 2; LAND = 3; FREEFLIGHT = 4
        self.drone_states = [DEFAULT, TAKEOFF, PAUSE, LAND, FREEFLIGHT]
        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                model=self.model, data=self.data, show_left_ui=True, show_right_ui=True
            )
        self.solver = QP(joint_idx=np.array([1, 2]), actuator_idx=np.array([6, 7]), 
                         λj=λj, λs=λs, λxy=λxy, λz=λz, λyaw=λyaw,
                         ps=ps, pz_l=pz_l, pz_h=pz_h,
                         λΣ=λΣ, tol=tol, ilimit=ilimit)
    
    def optimize(self, current_qpos: np.ndarray, ee_pos_target: np.ndarray, ee_quat_target: np.ndarray, drone_state: int = 0):
        # Set the drone state
        self.set_state(drone_state)
        # Calculate the forward kinematics (Tep) for the target end-effector pose
        Tep = calculate_arm_Te(ee_pos_target, ee_quat_target)
        self.data.qpos[:] = current_qpos.copy()
        result_IK = self.solver.solve(self.model, self.data, Tep)
        # Visualize the update
        if self.visualize: 
            # Update the target marker position
            self.model.site_pos[self.model.site("target_marker").id] = ee_pos_target
            self.data.qpos[:] = current_qpos.copy()
            mujoco.mj_fwdPosition(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.viewer.sync() # Sync the data from the model to the viewer
        km_target = np.zeros(9)
        if not result_IK[1]:
            print("Failed to find a solution, error: ", result_IK[4], "jl_valid: ", result_IK[5])
            km_target[0:3] = current_qpos[0:3]
            km_target[3:7] = current_qpos[3:7]
            km_target[7:9] = current_qpos[7:9]
        else:
            km_target[0:3] = result_IK[0][0:3]
            km_target[3:7] = result_IK[0][3:7]
            km_target[7:9] = result_IK[0][7:9]

        return km_target

    def set_state(self, drone_state: int):
        if drone_state not in self.drone_states:
            raise ValueError("Invalid drone state")
        if drone_state == 0: # DEFAULT: 1.0
            self.solver.λxy = 1.0
            self.solver.λz = 1.0
            self.solver.λyaw = 1.0
        elif drone_state == 1 or drone_state == 3: # TAKEOFF, LAND
            self.solver.λxy = 0.1
            # self.solver.λz = 0.1
            # self.solver.λyaw = 0.1
        elif drone_state == 4: # FREEFLIGHT (increase the arm movement)
            self.solver.λxy = 10.0
            self.solver.λz = 1.0
            self.solver.λyaw = 1.0

if __name__ == "__main__":
    # Test the IK planner
    xml_path = "/home/zyh/Aerial_Manipulation/sim/am_ws/src/trajectory_controller_uam/assets/hexa_scorpion.xml"
    planner = MjIKPlanner(xml_path)
    current_qpos = np.array([6.20448655e-05, -1.83004501e-11, 2.18594681e-01, 9.99999990e-01,
                            -2.93351747e-13, 1.41328819e-04, -1.94034843e-12, -6.42631903e-07,
                            4.83745156e-04, 0.00000000e+00, 0.00000000e+00])
    ee_pos_target = np.array([1.85122045e-01,-1.84596337e-11,3.21266681e-01])
    ee_quat_target = np.array([9.99999990e-01, 3.11133389e-13, 1.41328824e-04, -1.49916903e-12])
    TAKEOFF = 1; PAUSE = 2; LAND = 3; FREEFLIGHT = 4
    km_target = planner.optimize(current_qpos, ee_pos_target, ee_quat_target, TAKEOFF)
    print("km_target: ", km_target)