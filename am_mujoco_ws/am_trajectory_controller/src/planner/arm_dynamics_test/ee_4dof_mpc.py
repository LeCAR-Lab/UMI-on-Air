# mpc_file.py

import sys
sys.path.append('..')

import casadi as cs
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
sys.path.append('../..')
from planner.ik_util import rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca, rpy_to_rotation_matrix, quaternion_from_rotation_matrix
from scipy.spatial.transform import Rotation as R


DEGREE_TO_RADIAN = np.pi / 180.0


def quaternion_to_rpy_ca(quat):
    # quat is assumed to be a CasADi MX vector of shape (4,) with order [w, x, y, z]
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Compute roll, pitch, yaw
    roll  = cs.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = cs.asin(2*(w*y - z*x))
    yaw   = cs.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return cs.vertcat(roll, pitch, yaw)


def generate_dh_matrix_ca(alpha, a, d, theta):
    # Create a CasADi matrix for the transformation
    transform_matrix = ca.MX(4, 4)  # Initialize a 4x4 matrix

    # Fill in the matrix using CasADi's trigonometric functions
    transform_matrix[0, 0] = ca.cos(theta)
    transform_matrix[0, 1] = -ca.sin(theta) * ca.cos(alpha)
    transform_matrix[0, 2] = ca.sin(theta) * ca.sin(alpha)
    transform_matrix[0, 3] = a * ca.cos(theta)

    transform_matrix[1, 0] = ca.sin(theta)
    transform_matrix[1, 1] = ca.cos(theta) * ca.cos(alpha)
    transform_matrix[1, 2] = -ca.cos(theta) * ca.sin(alpha)
    transform_matrix[1, 3] = a * ca.sin(theta)

    transform_matrix[2, 0] = 0
    transform_matrix[2, 1] = ca.sin(alpha)
    transform_matrix[2, 2] = ca.cos(alpha)
    transform_matrix[2, 3] = d

    transform_matrix[3, 0] = 0
    transform_matrix[3, 1] = 0
    transform_matrix[3, 2] = 0
    transform_matrix[3, 3] = 1

    return transform_matrix

class DroneMPC_4DOF:
    def __init__(self, T, N, Q, R, R_delta, joint_min, joint_max, default_arm_angle):
        self.T = T
        self.N = N
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.R_delta = np.diag(R_delta)
        self.joint_min = np.array(joint_min) * DEGREE_TO_RADIAN
        self.joint_max = np.array(joint_max) * DEGREE_TO_RADIAN
        self.default_arm_angle = np.array(default_arm_angle) * DEGREE_TO_RADIAN
        self.arm_angle = cs.MX.sym('arm_angle', 3)
        self.arm_angle_ref = cs.MX.sym('arm_angle_ref', 3)
        self.ee_quat_ref = cs.MX.sym('ee_quat_ref', 4) # handle non-linear quat error
        self.u_prev = cs.MX.sym('u_prev', 3)
        self.param = cs.vertcat(self.ee_quat_ref, self.u_prev)
        self.x = cs.vertcat(self.arm_angle)
        self.u = cs.vertcat(self.arm_angle_ref)
        
        
        self.DH_params = np.array([[-np.pi/2, 0, 0, 0.0],
                            [-np.pi/2, 0.362895306, 0.0, -np.pi],
                            [-np.pi/2, 0.00711424939, 0.0496796518, 0.0],
                            [-np.pi/2, 0.441054359, 0.0, -np.pi],
                            [np.pi/2, 0.00980744858, 0.0762684723, 0.0],
                            [-np.pi/2, 0.0, 0.0, -np.pi/2]])
        self.arm_base_pos = np.array([0.0664397079, 0.0, -0.0171154472])
        self.ee_base_pos = np.array([0.149832002, -0.0189594673, -0.00617417526])

        
        
        self.x_dot = self.build_dynamics(self.x, self.u)
        self.model = self.build_acados_model()
        self.ocp_solver = self.build_acados_ocp_solver()
        
                

    def build_dynamics(self, x, u):
        arm_angle = x[0:3]
        arm_angle_ref = u[0:3]
        darm_angle = (arm_angle_ref - arm_angle) * 5.0
        return cs.vertcat(darm_angle)
    
    def arm_forward_kinematics(self, p, arm_angle, base_euler):
        joint_angles = cs.vertcat(0.0, arm_angle[0], 0.0, arm_angle[1], 0.0, arm_angle[2])
        
        T = cs.MX.eye(4)
        
        for i in range(6):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
            T = cs.mtimes(T, generate_dh_matrix_ca(alpha, a, d, theta))

        ee_pos = cs.mtimes(T, cs.vertcat(self.ee_base_pos, 1))
        ee_pos = ee_pos[0:3]/ee_pos[3]
        ee_ori_R = T[0:3, 0:3]
        
        base_ori_R = rotation_matrix_from_euler_ca(base_euler)
        
        ee_pos += self.arm_base_pos
        ee_pos = cs.mtimes(base_ori_R, ee_pos) + p
        ee_ori_R = cs.mtimes(base_ori_R, ee_ori_R)
        ee_quat = quaternion_from_rotation_matrix_ca(ee_ori_R)
        ee_state = cs.vertcat(ee_pos, ee_quat)
        return ee_state
    
    
    def build_cost_expr(self):
        p = cs.MX.zeros(3)
        base_euler = cs.MX.zeros(3)
        ee_state = self.arm_forward_kinematics(p, self.arm_angle, base_euler)
        ee_pos, ee_quat = ee_state[0:3], ee_state[3:7]
        ee_quat_ref = self.ee_quat_ref
        ee_euler_ref = quaternion_to_rpy_ca(ee_quat_ref)
        ee_euler = quaternion_to_rpy_ca(ee_quat)
        ee_pitch_error = (ee_euler[1] - ee_euler_ref[1])
        # ee_quat_error = 1.0 - cs.dot(ee_quat, ee_quat_ref)**2
        cost_expr_y = cs.vertcat(ee_pos, ee_pitch_error, self.arm_angle, self.u, self.u - self.u_prev)
        cost_expr_y_e = cs.vertcat(ee_pos, ee_pitch_error, self.arm_angle)
        return cost_expr_y, cost_expr_y_e
    

    def build_acados_model(self):
        x_dot = cs.MX.sym('x_dot', 3)
        f_expl = self.x_dot
        f_impl = x_dot - f_expl
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        
        cost_y_expr, cost_y_expr_e = self.build_cost_expr()
        model.cost_y_expr = cost_y_expr
        model.cost_y_expr_e = cost_y_expr_e
        
        
        model.x = self.x
        model.xdot = x_dot
        model.u = self.u
        model.p = self.param
        model.name = 'arm_4_dof'
        return model

    def build_acados_ocp_solver(self):
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.N
        ocp.dims.np = self.param.size()[0]
        ocp.parameter_values = np.zeros(ocp.dims.np)
        ocp.solver_options.tf = self.T
        nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = self.model.cost_y_expr.size()[0]
        ny_e = self.model.cost_y_expr_e.size()[0]
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # W = np.block([
        #     [self.Q, np.zeros((nx, nu))],
        #     [np.zeros((nu, nx)), self.R]
        # ])
        W = block_diag(self.Q, self.R, self.R_delta)
        ocp.cost.W = W
        ocp.cost.W_e = 10.0 * self.Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        
        x_min = np.concatenate([self.joint_min])
        x_max = np.concatenate([self.joint_max])
        u_min = np.concatenate([self.joint_min])
        u_max = np.concatenate([self.joint_max])

        ocp.constraints.lbx = x_min
        ocp.constraints.ubx = x_max
        ocp.constraints.idxbx = np.array(list(range(nx)))
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.array(list(range(nu)))
        p = cs.MX.zeros(3)
        base_euler = cs.MX.zeros(3)
        ee_state = self.arm_forward_kinematics(p, self.arm_angle, base_euler)
        ee_pos, ee_quat = ee_state[0:3], ee_state[3:7]
        ocp.model.con_h_expr = (ee_pos - (p + self.arm_base_pos))[2]
        ocp.constraints.lh = np.array([-0.1])
        ocp.constraints.uh = np.array([1.0])
        
        ocp.constraints.x0 = np.zeros(nx)
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.compile_dir = "./acados_mpc_build"
        
        # rebuild = False
        rebuild = True
        return AcadosOcpSolver(ocp, generate=rebuild, build=rebuild)

    def set_reference_sequence(self, ee_pos_refs, ee_quat_refs, base_euler, arm_angle, u_prev):
        u_ref = [0, 0, 0] # fxyz, arm_angle_ref
        default_arm_angle = np.array([self.default_arm_angle[0], self.default_arm_angle[1], arm_angle[2]])
        for i in range(self.N):
            x_ref = np.concatenate([ee_pos_refs[i], np.zeros(1), default_arm_angle])
            ref = np.concatenate([x_ref, u_ref, np.zeros(self.R_delta.shape[0])])
            params = np.concatenate([ee_quat_refs[i], u_prev])
            self.ocp_solver.set(i, "yref", ref)
            self.ocp_solver.set(i, "p", params)
        final_x_ref = np.concatenate([ee_pos_refs[-1], np.zeros(1), default_arm_angle])
        params = np.concatenate([ee_quat_refs[-1], u_prev])
        self.ocp_solver.set(self.N, "yref", final_x_ref)
        self.ocp_solver.set(self.N, "p", params)

    def run_optimization(self, x0):
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.solve()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
        return u_opt, x_opt


class ArmMPCPlanner:
    def __init__(self, T, N, Q, R, R_delta, joint_min, joint_max, default_arm_angle, output_filter_gain):
        self.mpc = DroneMPC_4DOF(T, N, Q, R, R_delta, joint_min, joint_max, default_arm_angle)
        self.x0 = np.zeros(3)
        self.output_filter_gain = np.array(output_filter_gain)
        self.u_cmd_last = None

    def optimize(self, arm_angle, base_euler, ee_pos_refs, ee_quat_refs, u_prev):
        self.x0 = np.concatenate([arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, base_euler, arm_angle, u_prev)
        u_opt, x_opt = self.mpc.run_optimization(self.x0)
        u_cmd = u_opt[0]
        # if self.u_cmd_last is None:
        #     self.u_cmd_last = u_cmd
        # else:
        #     u_cmd = self.output_filter_gain * u_cmd + (1 - self.output_filter_gain) * self.u_cmd_last
        #     self.u_cmd_last = u_cmd
        arm_angle_opt = x_opt[:, 0:3]
        arm_angle_cmd = u_cmd[0:3]    
        
        return arm_angle_opt[0], arm_angle_cmd



    def forward_kinematics(self, base_pos, base_euler, arm_angle):
        '''
            input: u: [base_pos, base_euler(rpy), joint_angles]
            output: ee_state: [ee_pos, ee_quat]
        '''

        joint_angles = np.array([0.0, arm_angle[0], 0.0, arm_angle[1], 0.0,  arm_angle[2]])


        T = np.eye(4)

        for i in range(6):
            DH_params = self.mpc.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
            dh_matrix = self.generate_dh_matrix(alpha, a, d, theta)
            T = np.dot(T, dh_matrix)

        ee_pos = np.array([self.mpc.ee_base_pos[0], self.mpc.ee_base_pos[1], self.mpc.ee_base_pos[2], 1])
        ee_pos = np.dot(T, ee_pos)
        ee_pos = ee_pos[:3]/ee_pos[3]
        ee_orientation_matrix = T[:3, :3]
        

        base_rotation_matrix = rpy_to_rotation_matrix(base_euler)
        ee_pos = ee_pos + self.mpc.arm_base_pos
        ee_pos = np.dot(base_rotation_matrix, ee_pos[:3])
        ee_pos = ee_pos + base_pos
        ee_orientation_matrix = np.dot(base_rotation_matrix, ee_orientation_matrix)
        ee_quat = quaternion_from_rotation_matrix(ee_orientation_matrix)
        ee_state = np.concatenate((ee_pos, ee_quat))

        return ee_state
    
    
    def generate_dh_matrix(self, alpha, a, d, theta):

        transform_matrix = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                            [0, np.sin(alpha), np.cos(alpha), d],
                            [0, 0, 0, 1]])

        return transform_matrix
