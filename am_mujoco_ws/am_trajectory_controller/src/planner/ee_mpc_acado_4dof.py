# mpc_file.py

import sys
sys.path.append('..')

import casadi as cs
import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from planner.ik_util import rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca, rpy_to_rotation_matrix, quaternion_from_rotation_matrix
from scipy.spatial.transform import Rotation as R
import os


DEGREE_TO_RADIAN = np.pi / 180.0


def quaternion_to_rpy_ca(quat):
    # quat is assumed to be a CasADi MX vector of shape (4,) with order [w, x, y, z]
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # Compute roll, pitch, yaw
    roll  = cs.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = cs.asin(2*(w*y - z*x))
    yaw   = cs.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return cs.vertcat(roll, pitch, yaw)


class DroneMPC_4DOF:
    def __init__(self, mass, T, N, N_mini, Q, R, R_delta, acc2thrust_gain, pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max, default_arm_angle, moment_of_inertia, compile_dir="./acados_mpc_build", model_name="fully_actuated_uav", fix_reference_quat=False):
        self.mass = mass
        self.T = T
        self.N = N
        self.N_mini = N_mini
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.R_delta = np.diag(R_delta)
        self.acc2thrust_gain = np.array(acc2thrust_gain)
        self.pos_min = np.array(pos_min)
        self.pos_max = np.array(pos_max)
        self.vel_min = np.array(vel_min)
        self.vel_max = np.array(vel_max)
        self.acc_min = np.array(acc_min)
        self.acc_max = np.array(acc_max)
        self.joint_min = np.array(joint_min) * DEGREE_TO_RADIAN
        self.joint_max = np.array(joint_max) * DEGREE_TO_RADIAN
        self.default_arm_angle = np.array(default_arm_angle) * DEGREE_TO_RADIAN
        self.arm_delay_gain = 12.0
        self.compile_dir = compile_dir
        self.model_name = model_name  # Store the unique model name
        self.fix_reference_quat = fix_reference_quat  # Flag to fix reference quaternion
        print(f"DEBUG: DroneMPC_4DOF constructor - fix_reference_quat = {fix_reference_quat}")
        self.fixed_quat = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])  # Fixed quaternion [w, x, y, z] - 90° rotation around X-axis
        
        if N == N_mini: self.mini_mpc = True
        else: self.mini_mpc = False
        
        # Moment of inertia
        self.I_x = moment_of_inertia[0]
        self.I_y = moment_of_inertia[1] 
        self.I_z = moment_of_inertia[2]

        # State vector: [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)] - 16 states
        self.x = cs.MX.sym('x', 16)
        self.p = self.x[0:3]
        self.v = self.x[3:6]
        self.base_euler = self.x[6:9]
        self.base_ang_vel = self.x[9:12]  # New angular velocity states
        self.arm_angle = self.x[12:16]

        # Control vector: [acc(3), torques(3), arm_angle_ref(4)] - 10 controls
        self.u = cs.MX.sym('u', 10)
        self.u_prev = cs.MX.sym('u_prev', 10)

        # Parameters for reference trajectory
        self.ee_pos_ref = cs.MX.sym('ee_pos_ref', 3)
        self.ee_quat_ref = cs.MX.sym('ee_quat_ref', 4)
        self.arm_angle_ref = cs.MX.sym('arm_angle_ref', 4)
        self.param = cs.vertcat(self.ee_pos_ref, self.ee_quat_ref, self.arm_angle_ref, self.u_prev)

        self.x_dot = self.build_dynamics(self.x, self.u)
        self.model = self.build_acados_model()
        self.ocp_solver = self.build_acados_ocp_solver()
        
        # ------------------------------------------------------------------
        # Build helper expressions for tracking-only cost *and* its gradient
        # w.r.t. the reference end-effector position (first 3 elements of p).
        # This does NOT influence optimisation; it is used purely after a
        # solve for diagnostics or learning signals.
        # ------------------------------------------------------------------
        self._build_tracking_helpers()

    def build_dynamics(self, x, u):
        p = x[0:3]
        v = x[3:6]
        base_euler = x[6:9]
        base_ang_vel = x[9:12]  # Angular velocities
        arm_angle = x[12:16]

        acc = u[0:3]
        torques = u[3:6]
        arm_angle_ref = u[6:10]

        dp = v
        dv = acc / self.mass
        
        # Proper dynamics: euler rates from angular velocities
        d_base_euler = base_ang_vel
        
        # Angular acceleration from torques (this is what provides control)
        d_base_ang_vel = cs.vertcat(torques[0] / self.I_x, torques[1] / self.I_y, torques[2] / self.I_z)
        
        darm_angle = (arm_angle_ref - arm_angle) * self.arm_delay_gain

        return cs.vertcat(dp, dv, d_base_euler, d_base_ang_vel, darm_angle)
    
    def arm_forward_kinematics(self, p, arm_angle, base_euler):
        # Use direct transformation chain matching MuJoCo XML structure (same as IK implementation)
        joint_angles = arm_angle  # [joint1, joint2, joint3, joint4]

        # Base rotation matrix
        base_ori_R = rotation_matrix_from_euler_ca(base_euler)
        
        # Build transformation chain exactly as in MuJoCo XML:
        # 1. base_link to arm_base_link: pos="0.088 0 0.0"
        pos_arm_base = cs.vertcat(0.088, 0.0, 0.0)
        
        # 2. arm_base_link to manipulation_link1_pitch_link: pos="0.0 0.0 0.06475"
        pos_link1 = pos_arm_base + cs.vertcat(0.0, 0.0, 0.06475)
        
        # 3. Apply joint1 rotation (Y-axis, pitch): axis="0 1 0"
        c1, s1 = cs.cos(joint_angles[0]), cs.sin(joint_angles[0])
        rot_joint1 = cs.vertcat(
            cs.horzcat(c1, 0, s1),
            cs.horzcat(0, 1, 0),
            cs.horzcat(-s1, 0, c1)
        )
        
        # 4. manipulation_link1 to manipulation_link2: pos="-0.3795 0.0 0.059" quat="0.707 0.707 0 0"
        # The quat="0.707 0.707 0 0" represents a 90° rotation around X-axis
        pos_link2_local = cs.vertcat(-0.3795, 0.0, 0.059)
        pos_link2 = pos_link1 + cs.mtimes(rot_joint1, pos_link2_local)
        
        # Link2 has initial rotation quat="0.707 0.707 0 0" = 90° around X
        rot_link2_initial = cs.vertcat(
            cs.horzcat(1, 0, 0),
            cs.horzcat(0, 0, -1),
            cs.horzcat(0, 1, 0)
        )
        
        # 5. Apply joint2 rotation (Z-axis, yaw): axis="0 0 1"  
        c2, s2 = cs.cos(joint_angles[1]), cs.sin(joint_angles[1])
        rot_joint2 = cs.vertcat(
            cs.horzcat(c2, -s2, 0),
            cs.horzcat(s2, c2, 0),
            cs.horzcat(0, 0, 1)
        )
        
        # Combined rotation for link2
        rot_link2 = cs.mtimes(cs.mtimes(rot_joint1, rot_link2_initial), rot_joint2)
        
        # 6. manipulation_link2 to manipulation_link3: pos="0.4475 0 0"
        pos_link3_local = cs.vertcat(0.4475, 0.0, 0.0)
        pos_link3 = pos_link2 + cs.mtimes(rot_link2, pos_link3_local)
        
        # 7. Apply joint3 rotation (Z-axis, yaw): axis="0 0 1"
        c3, s3 = cs.cos(joint_angles[2]), cs.sin(joint_angles[2])
        rot_joint3 = cs.vertcat(
            cs.horzcat(c3, -s3, 0),
            cs.horzcat(s3, c3, 0),
            cs.horzcat(0, 0, 1)
        )
        
        rot_link3 = cs.mtimes(rot_link2, rot_joint3)
        
        # 8. manipulation_link3 to manipulation_link4: pos="0.071 0 0"
        pos_link4_local = cs.vertcat(0.071, 0.0, 0.0)
        pos_link4 = pos_link3 + cs.mtimes(rot_link3, pos_link4_local)
        
        # 9. Apply joint4 rotation (negative X-axis, roll): axis="-1 0 0"
        c4, s4 = cs.cos(joint_angles[3]), cs.sin(joint_angles[3])
        rot_joint4 = cs.vertcat(
            cs.horzcat(1, 0, 0),
            cs.horzcat(0, c4, s4),    # Note: positive because axis is "-1 0 0"
            cs.horzcat(0, -s4, c4)
        )
        
        rot_link4 = cs.mtimes(rot_link3, rot_joint4)
        
        # 10. manipulation_link4 to ee: pos="0.01 0 0"
        # 11. ee to ee_tool: pos="0.14 0 0"
        # Total: 0.01 + 0.14 = 0.15
        pos_ee_local = cs.vertcat(0.01, 0.0, 0.0)
        pos_ee = pos_link4 + cs.mtimes(rot_link4, pos_ee_local)
        
        # Apply base rotation and translation
        ee_pos_world = cs.mtimes(base_ori_R, pos_ee) + p
        ee_ori_R_world = cs.mtimes(base_ori_R, rot_link4)
        
        # Convert rotation matrix to quaternion
        ee_quat = quaternion_from_rotation_matrix_ca(ee_ori_R_world)
        
        ee_state = cs.vertcat(ee_pos_world, ee_quat)
        return ee_state
    
    
    def build_cost_expr(self):
        # Use yaw from state for building base_euler
        base_euler_state = self.base_euler
        ee_state = self.arm_forward_kinematics(self.p, self.arm_angle, base_euler_state)
        ee_pos, ee_quat = ee_state[0:3], ee_state[3:7]
        ee_quat_ref = self.ee_quat_ref
        ee_euler_ref = quaternion_to_rpy_ca(ee_quat_ref)
        ee_euler = quaternion_to_rpy_ca(ee_quat)
        ee_roll_error = ee_euler[0] - ee_euler_ref[0]
        ee_pitch_error = ee_euler[1] - ee_euler_ref[1]
        ee_yaw_error   = ee_euler[2] - ee_euler_ref[2]
        ee_pos_error = ee_pos - self.ee_pos_ref
        # Include base_euler and angular velocities in the cost expression
        cost_expr_y = cs.vertcat(ee_pos_error, ee_roll_error, ee_pitch_error, ee_yaw_error, self.v, self.base_euler, self.base_ang_vel, self.arm_angle, self.u, self.u - self.u_prev)
        cost_expr_y_e = cs.vertcat(ee_pos_error, ee_roll_error, ee_pitch_error, ee_yaw_error, self.v, self.base_euler, self.base_ang_vel, self.arm_angle)
        return cost_expr_y, cost_expr_y_e
    

    def build_acados_model(self):
        x_dot = cs.MX.sym('x_dot', 16)
        f_expl = self.x_dot
        f_impl = x_dot - f_expl
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        
        cost_y_expr, cost_y_expr_e = self.build_cost_expr()
        self.cost_y_expr    = cost_y_expr
        self.cost_y_expr_e  = cost_y_expr_e
        
        model.cost_y_expr = cost_y_expr
        model.cost_y_expr_e = cost_y_expr_e
        
        model.x = self.x
        model.xdot = x_dot
        model.u = self.u
        model.p = self.param
        model.name = self.model_name 
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
        W = block_diag(self.Q, self.R, self.R_delta)
        ocp.cost.W = W
        
        # Terminal cost weight matrix - only for the state terms in cost_y_expr_e
        # cost_y_expr_e has: [ee_pos(3), ee_roll_error, ee_pitch_error, ee_yaw_error, v(3), base_euler(3), base_ang_vel(3), arm_angle(4)]
        # Total: 3 + 3 + 3 + 3 + 3 + 4 = 19 dimensions
        # Extract only the state-related weights from Q (first 19 elements)
        Q_terminal = self.Q[:19, :19]  # Take only the state-related part of Q
        ocp.cost.W_e = 10.0 * Q_terminal
        
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)
        
        # State constraints including base orientation and angular velocities
        euler_min = np.array([-0.5, -0.5, -np.pi]) 
        euler_max = np.array([0.5, 0.5, np.pi])  
        ang_vel_min = np.array([-10.0, -10.0, -10.0])  # Angular velocity limits (rad/s)
        ang_vel_max = np.array([10.0, 10.0, 10.0])
        x_min = np.concatenate([self.pos_min, self.vel_min, euler_min, ang_vel_min, self.joint_min])
        x_max = np.concatenate([self.pos_max, self.vel_max, euler_max, ang_vel_max, self.joint_max])

        # Control constraints: forces + torques + arm angles
        torque_min, torque_max = -20.0, 20.0
        u_min = np.concatenate([self.acc_min * self.mass, [torque_min]*3, self.joint_min])
        u_max = np.concatenate([self.acc_max * self.mass, [torque_max]*3, self.joint_max])

        ocp.constraints.lbx = x_min
        ocp.constraints.ubx = x_max
        ocp.constraints.idxbx = np.array(list(range(nx)))
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.array(list(range(nu)))
        
        # Path constraints (EE height above base)
        base_euler_state = self.base_euler
        ee_state = self.arm_forward_kinematics(self.p, self.arm_angle, base_euler_state)
        ee_pos, ee_quat = ee_state[0:3], ee_state[3:7]
        h_expr = cs.vertcat((ee_pos - self.p)[2])  # EE height above base
        
        ocp.model.con_h_expr = h_expr
        
        ocp.constraints.lh = np.array([-0.1])
        ocp.constraints.uh = np.array([1.0])
        
        ocp.constraints.x0 = np.zeros(nx)
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        
        # Add iteration limits and tolerances to prevent QP solver errors
        ocp.solver_options.qp_solver_iter_max = 1000  # Increase QP solver iteration limit
        ocp.solver_options.nlp_solver_max_iter = 100  # Set NLP solver iteration limit
        ocp.solver_options.qp_solver_tol_stat = 1e-6  # QP solver stationarity tolerance
        ocp.solver_options.qp_solver_tol_eq = 1e-6    # QP solver equality constraint tolerance
        ocp.solver_options.qp_solver_tol_ineq = 1e-6  # QP solver inequality constraint tolerance
        ocp.solver_options.qp_solver_tol_comp = 1e-6  # QP solver complementarity tolerance
        ocp.solver_options.qp_solver_warm_start = 0   # Enable warm starting
        
        ocp.solver_options.compile_dir = self.compile_dir
        
        rebuild = True  # Always rebuild to ensure correct dimensions
        # if not os.path.exists(ocp.solver_options.compile_dir):
        #     rebuild = True

        return AcadosOcpSolver(ocp, generate=rebuild, build=rebuild)

    def set_reference_sequence(self, ee_pos_refs, ee_quat_refs, arm_angle, u_prev):
        # Store references for tracking cost computation
        self._ee_pos_refs = ee_pos_refs.copy()
        self._ee_quat_refs = ee_quat_refs.copy()
        
        # Override quaternion references if flag is enabled
        if self.fix_reference_quat:
            self._ee_quat_refs = np.tile(self.fixed_quat, (len(ee_quat_refs), 1))
            ee_quat_refs = self._ee_quat_refs
        
        default_arm_angle = np.array([self.default_arm_angle[0], self.default_arm_angle[1], self.default_arm_angle[2], self.default_arm_angle[3]])
        for i in range(self.N):
            # Extract desired yaw from reference quaternion
            r = R.from_quat([ee_quat_refs[i][1], ee_quat_refs[i][2], ee_quat_refs[i][3], ee_quat_refs[i][0]])
            ee_euler_ref = r.as_euler('xyz')
            base_euler_ref = np.array([0, 0, ee_euler_ref[2]]) # Keep base roll and pitch at 0

            # Control reference: zero forces/torque and default arm angles
            u_ref = np.zeros(10)
            u_ref[6:10] = default_arm_angle

            # Cost reference matching the cost expression order:
            # [ee_pos_error(3), ee_roll_error, ee_pitch_error, ee_yaw_error, v(3), base_euler(3), base_ang_vel(3), arm_angle(4), u(10), u_delta(10)]
            cost_ref = np.concatenate([
                np.zeros(3),          # zero EE position error (3)
                np.zeros(3),          # desired roll, pitch & yaw errors = 0 (3)
                np.zeros(3),          # desired base velocity = 0 (3)
                base_euler_ref,       # desired base euler (3)
                np.zeros(3),          # desired base angular velocity = 0 (3)
                default_arm_angle,    # desired arm angles (4)
                u_ref,                # control reference (10)
                np.zeros(10)          # zero control delta reference (10)
            ])
            
            params = np.concatenate([ee_pos_refs[i], ee_quat_refs[i], default_arm_angle, u_prev])
            self.ocp_solver.set(i, "yref", cost_ref)
            self.ocp_solver.set(i, "p", params)

        # final node - terminal cost matching cost_expr_y_e order:
        # [ee_pos(3), ee_roll_error, ee_pitch_error, ee_yaw_error, v(3), base_euler(3), base_ang_vel(3), arm_angle(4)]
        r = R.from_quat([ee_quat_refs[-1][1], ee_quat_refs[-1][2], ee_quat_refs[-1][3], ee_quat_refs[-1][0]])
        ee_euler_ref_final = r.as_euler('xyz')
        base_euler_ref_final = np.array([0, 0, ee_euler_ref_final[2]])
        final_cost_ref = np.concatenate([
            np.zeros(3),          # zero EE position error (3)
            np.zeros(3),          # desired roll, pitch & yaw errors = 0 (3)
            np.zeros(3),          # desired base velocity = 0 (3)
            base_euler_ref_final, # desired base euler (3)
            np.zeros(3),          # desired base angular velocity = 0 (3)
            default_arm_angle,    # desired arm angles (4)
        ])
        params = np.concatenate([ee_pos_refs[-1], ee_quat_refs[-1], default_arm_angle, u_prev])
        self.ocp_solver.set(self.N, "yref", final_cost_ref)
        self.ocp_solver.set(self.N, "p", params)

    def run_optimization(self, x0):
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.solve()
        total_cost = self.ocp_solver.get_cost()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
        return u_opt, x_opt, total_cost

    def run_optimization_with_tracking_cost(self, x0):
        """Run optimization and return both total cost and tracking-only cost"""
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.solve()
        
        total_cost = self.ocp_solver.get_cost()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
        
        # Extract tracking-only cost using Acados residuals
        tracking_cost = self._extract_tracking_cost()
        
        return u_opt, x_opt, total_cost, tracking_cost
    
    def run_optimization_with_tracking_gradient(self, x0):
        """Run optimization and return both total cost and tracking-only cost"""
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.solve()
        
        total_cost = self.ocp_solver.get_cost()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
        
        # Extract tracking-only cost using Acados residuals
        tracking_cost = self._extract_tracking_cost()
        tracking_gradient = self._extract_tracking_gradients()
        
        return u_opt, x_opt, total_cost, tracking_cost, tracking_gradient
    
    def _extract_tracking_cost(self):
        """Compute EE tracking cost using pre-defined cost functions."""
        dt = self.T / self.N
        trk_cost = 0.0

        for k in range(self.N):
            xk = self.ocp_solver.get(k, "x")
            uk = self.ocp_solver.get(k, "u")
            pk = self.ocp_solver.get(k, "p")

            cost_k = float(self._trk_stage_cost_func(xk, uk, pk))
            trk_cost += dt * cost_k

        # Terminal node (no dt scaling)
        xN = self.ocp_solver.get(self.N, "x")
        pN = self.ocp_solver.get(self.N, "p")
        cost_N = float(self._trk_term_cost_func(xN, pN))
        trk_cost += cost_N

        return trk_cost

    def get_tracking_cost(self):
        """Get tracking cost from last optimization (call after run_optimization)"""
        return self._extract_tracking_cost()

    def _build_tracking_helpers(self):
        """Create CasADi functions to (1) evaluate tracking error vector and
        (2) compute the gradient of stage / terminal tracking cost w.r.t.
        the reference end-effector *position and quaternion* contained in the parameter
        vector.  By default we use the first six components of y (EE pos +
        orientation error) and the corresponding slice of Q.
        """
        # Indices of y that constitute the tracking error (EE position + ori)
        self.trk_idx = np.arange(6)

        # Build tracking error vectors now that EE position error is directly in cost_y_expr.
        pos_err_stage = self.cost_y_expr[0:3]
        pos_err_term  = self.cost_y_expr_e[0:3]

        ori_err_stage = self.cost_y_expr[3:6]   # already errors (roll,pitch,yaw)
        ori_err_term  = self.cost_y_expr_e[3:6]

        trk_err_stage = ca.vertcat(pos_err_stage, ori_err_stage)
        trk_err_term  = ca.vertcat(pos_err_term,  ori_err_term)

        # Corresponding weight matrices (slice of Q / Q_e)
        Q_trk  = self.Q[np.ix_(self.trk_idx, self.trk_idx)]
        Qe_trk = 10.0 * Q_trk
        self._Q_trk  = Q_trk
        self._Qe_trk = Qe_trk

        trk_stage_cost_expr = 0.5 * ca.mtimes([trk_err_stage.T, ca.DM(Q_trk), trk_err_stage])
        trk_term_cost_expr  = 0.5 * ca.mtimes([trk_err_term.T,  ca.DM(Qe_trk), trk_err_term])

        # Wrap as functions to evaluate tracking cost quickly
        self._trk_stage_cost_func = ca.Function(
            "stage_trk_cost", [self.x, self.u, self.param], [trk_stage_cost_expr]
        )
        self._trk_term_cost_func = ca.Function(
            "term_trk_cost", [self.x, self.param], [trk_term_cost_expr]
        )

        # Gradients w.r.t. both reference EE position (first 3 of param) and quaternion (next 4 of param)
        trk_grad_pos_stage_expr = ca.gradient(trk_stage_cost_expr, self.ee_pos_ref)
        trk_grad_quat_stage_expr = ca.gradient(trk_stage_cost_expr, self.ee_quat_ref)
        trk_grad_combined_stage_expr = ca.vertcat(trk_grad_pos_stage_expr, trk_grad_quat_stage_expr)

        trk_grad_pos_term_expr = ca.gradient(trk_term_cost_expr, self.ee_pos_ref)
        trk_grad_quat_term_expr = ca.gradient(trk_term_cost_expr, self.ee_quat_ref)
        trk_grad_combined_term_expr = ca.vertcat(trk_grad_pos_term_expr, trk_grad_quat_term_expr)

        # Wrap as CasADi functions for fast numerical evaluation post-solve
        self._trk_grad_stage_func = ca.Function(
            "grad_stage_trk", [self.x, self.u, self.param], [trk_grad_combined_stage_expr]
        )
        self._trk_grad_term_func = ca.Function(
            "grad_term_trk",  [self.x, self.param],        [trk_grad_combined_term_expr]
        )

    # ------------------------------------------------------------------
    # Public helpers to evaluate tracking diagnostics post optimisation
    # ------------------------------------------------------------------
    def _extract_tracking_gradients(self):
        """Return dJ/d(ee_pos_ref, ee_quat_ref) for every *stage* knot.

        Output shape: (N, 7) - combined position and quaternion gradients per time step.
        Format: [pos_grad(3), quat_grad(4)] where quat_grad is w.r.t. [w, x, y, z].
        """
        dt = self.T / self.N
        stage_grads = []

        # Stage costs (k=0 to N-1)
        for k in range(self.N):
            xk = self.ocp_solver.get(k, "x")
            uk = self.ocp_solver.get(k, "u")
            pk = self.ocp_solver.get(k, "p")
            gk = self._trk_grad_stage_func(xk, uk, pk).full().squeeze()
            stage_grads.append(gk * dt)

        # Stack into (N, 7) and return directly
        return np.stack(stage_grads, axis=0)

    def reset(self, reset_qp_solver_mem=1):
        """Reset the MPC solver to initial state (all zeros)"""
        self.ocp_solver.reset(reset_qp_solver_mem)


class ArmMPCPlanner:
    def __init__(self, mass, T, N, N_mini, Q, R, R_delta, acc2thrust_gain, pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max, default_arm_angle, output_filter_gain, moment_of_inertia, compile_dir="./acados_mpc_build", model_name="fully_actuated_uav", fix_reference_quat=False):
        self.mpc = DroneMPC_4DOF(mass, T, N, N_mini, Q, R, R_delta, acc2thrust_gain, pos_min, pos_max, vel_min, vel_max, acc_min, acc_max, joint_min, joint_max, default_arm_angle, moment_of_inertia, compile_dir, model_name, fix_reference_quat)
        self.x0 = np.zeros(16)  # State: p, v, base_euler, base_ang_vel, arm_angle
        self.output_filter_gain = np.array(output_filter_gain)
        self.u_cmd_last = None

    def optimize(self, p, v, arm_angle, base_euler, base_ang_vel, ee_pos_refs, ee_quat_refs, u_prev):
        # State vector: [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)] - 16 states
        # Use the provided angular velocity from the environment
        self.x0 = np.concatenate([p, v, base_euler, base_ang_vel, arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
        u_opt, x_opt, total_cost = self.mpc.run_optimization(self.x0)
        u_cmd = u_opt[0]
        if self.u_cmd_last is None:
            self.u_cmd_last = u_cmd
        else:
            u_cmd = self.output_filter_gain * u_cmd + (1 - self.output_filter_gain) * self.u_cmd_last
            self.u_cmd_last = u_cmd
        p_opt = x_opt[:, 0:3]
        v_opt = x_opt[:, 3:6]
        base_euler_opt = x_opt[:, 6:9]
        base_ang_vel_opt = x_opt[:, 9:12]  # New angular velocity states
        arm_angle_opt = x_opt[:, 12:16]    # Updated indexing for arm angles
        force_opt = u_cmd[0:3]
        torque_cmd = u_cmd[3:6]
        arm_angle_cmd = u_cmd[6:10]
        force_opt = force_opt * self.mpc.acc2thrust_gain
        
        force_opt = np.clip(force_opt, self.mpc.acc_min * self.mpc.mass * self.mpc.acc2thrust_gain, self.mpc.acc_max * self.mpc.mass * self.mpc.acc2thrust_gain)
        arm_angle_cmd = np.clip(arm_angle_cmd, self.mpc.joint_min, self.mpc.joint_max)
        arm_angle_delta = arm_angle_cmd - arm_angle
        arm_vel_limit = np.array([5.0, 5.0, 10.0, 10.0]) * DEGREE_TO_RADIAN
        arm_angle_delta = np.clip(arm_angle_delta, -arm_vel_limit, arm_vel_limit)
        arm_angle_cmd = arm_angle + arm_angle_delta
        
        return force_opt, torque_cmd, p_opt[-1], v_opt[-1], arm_angle_opt[-1], arm_angle_cmd, base_euler_opt[-1], total_cost

    def optimize_with_tracking_cost(self, p, v, arm_angle, base_euler, base_ang_vel, ee_pos_refs, ee_quat_refs, u_prev):
        """Optimize and return both total cost and tracking-only cost"""
        # State vector: [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)] - 16 states
        self.x0 = np.concatenate([p, v, base_euler, base_ang_vel, arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
        u_opt, x_opt, total_cost, tracking_cost = self.mpc.run_optimization_with_tracking_cost(self.x0)
        
        u_cmd = u_opt[0]
        if self.u_cmd_last is None:
            self.u_cmd_last = u_cmd
        else:
            u_cmd = self.output_filter_gain * u_cmd + (1 - self.output_filter_gain) * self.u_cmd_last
            self.u_cmd_last = u_cmd
        
        p_opt = x_opt[:, 0:3]
        v_opt = x_opt[:, 3:6]
        base_euler_opt = x_opt[:, 6:9]
        base_ang_vel_opt = x_opt[:, 9:12]
        arm_angle_opt = x_opt[:, 12:16]
        force_opt = u_cmd[0:3]
        torque_cmd = u_cmd[3:6]
        arm_angle_cmd = u_cmd[6:10]
        force_opt = force_opt * self.mpc.acc2thrust_gain
        
        force_opt = np.clip(force_opt, self.mpc.acc_min * self.mpc.mass * self.mpc.acc2thrust_gain, self.mpc.acc_max * self.mpc.mass * self.mpc.acc2thrust_gain)
        arm_angle_cmd = np.clip(arm_angle_cmd, self.mpc.joint_min, self.mpc.joint_max)
        arm_angle_delta = arm_angle_cmd - arm_angle
        arm_vel_limit = np.array([10.0, 10.0, 30.0, 30.0]) * DEGREE_TO_RADIAN
        arm_angle_delta = np.clip(arm_angle_delta, -arm_vel_limit, arm_vel_limit)
        arm_angle_cmd = arm_angle + arm_angle_delta
        
        # Convert MPC state trajectory to EE poses
        predicted_ee_trajectory = self._convert_x_opt_to_ee_poses(x_opt)
        
        return force_opt, torque_cmd, p_opt[-1], v_opt[-1], arm_angle_opt[-1], arm_angle_cmd, base_euler_opt[-1], total_cost, tracking_cost, predicted_ee_trajectory
    
    def optimize_with_tracking_gradient(self, p, v, arm_angle, base_euler, base_ang_vel, ee_pos_refs, ee_quat_refs, u_prev):
        """Optimize and return both total cost and tracking-only cost"""
        # State vector: [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)] - 16 states
        self.x0 = np.concatenate([p, v, base_euler, base_ang_vel, arm_angle])
        self.mpc.set_reference_sequence(ee_pos_refs, ee_quat_refs, arm_angle, u_prev)
        u_opt, x_opt, total_cost, tracking_cost, tracking_gradient = self.mpc.run_optimization_with_tracking_gradient(self.x0)
        
        return total_cost, tracking_cost, tracking_gradient, x_opt

    def forward_kinematics(self, base_pos, base_euler, arm_angle):
        '''
            input: base_pos, base_euler(rpy), arm_angle (4 DOF)
            output: ee_state: [ee_pos, ee_quat]
        '''
        # Use direct transformation chain matching MuJoCo XML structure (same as MPC implementation)
        joint_angles = arm_angle  # [joint1, joint2, joint3, joint4]

        # Base rotation matrix
        base_rotation_matrix = rpy_to_rotation_matrix(base_euler)
        
        # Build transformation chain exactly as in MuJoCo XML:
        # 1. base_link to arm_base_link: pos="0.088 0 0.0"
        pos_arm_base = np.array([0.088, 0.0, 0.0])
        
        # 2. arm_base_link to manipulation_link1_pitch_link: pos="0.0 0.0 0.06475"
        pos_link1 = pos_arm_base + np.array([0.0, 0.0, 0.06475])
        
        # 3. Apply joint1 rotation (Y-axis, pitch): axis="0 1 0"
        c1, s1 = np.cos(joint_angles[0]), np.sin(joint_angles[0])
        rot_joint1 = np.array([[c1, 0, s1],
                               [0, 1, 0],
                               [-s1, 0, c1]])
        
        # 4. manipulation_link1 to manipulation_link2: pos="-0.3795 0.0 0.059" quat="0.707 0.707 0 0"
        # The quat="0.707 0.707 0 0" represents a 90° rotation around X-axis
        pos_link2_local = np.array([-0.3795, 0.0, 0.059])
        pos_link2 = pos_link1 + np.dot(rot_joint1, pos_link2_local)
        
        # Link2 has initial rotation quat="0.707 0.707 0 0" = 90° around X
        rot_link2_initial = np.array([[1, 0, 0],
                                      [0, 0, -1],
                                      [0, 1, 0]])
        
        # 5. Apply joint2 rotation (Z-axis, yaw): axis="0 0 1"  
        c2, s2 = np.cos(joint_angles[1]), np.sin(joint_angles[1])
        rot_joint2 = np.array([[c2, -s2, 0],
                               [s2, c2, 0],
                               [0, 0, 1]])
        
        # Combined rotation for link2
        rot_link2 = np.dot(np.dot(rot_joint1, rot_link2_initial), rot_joint2)
        
        # 6. manipulation_link2 to manipulation_link3: pos="0.4475 0 0"
        pos_link3_local = np.array([0.4475, 0.0, 0.0])
        pos_link3 = pos_link2 + np.dot(rot_link2, pos_link3_local)
        
        # 7. Apply joint3 rotation (Z-axis, yaw): axis="0 0 1"
        c3, s3 = np.cos(joint_angles[2]), np.sin(joint_angles[2])
        rot_joint3 = np.array([[c3, -s3, 0],
                               [s3, c3, 0],
                               [0, 0, 1]])
        
        rot_link3 = np.dot(rot_link2, rot_joint3)
        
        # 8. manipulation_link3 to manipulation_link4: pos="0.071 0 0"
        pos_link4_local = np.array([0.071, 0.0, 0.0])
        pos_link4 = pos_link3 + np.dot(rot_link3, pos_link4_local)
        
        # 9. Apply joint4 rotation (negative X-axis, roll): axis="-1 0 0"
        c4, s4 = np.cos(joint_angles[3]), np.sin(joint_angles[3])
        rot_joint4 = np.array([[1, 0, 0],
                               [0, c4, s4],    # Note: positive because axis is "-1 0 0"
                               [0, -s4, c4]])
        
        rot_link4 = np.dot(rot_link3, rot_joint4)
        
        # 10. manipulation_link4 to ee: pos="0.01 0 0"
        # 11. ee to ee_tool: pos="0.14 0 0"
        # Total: 0.01 + 0.14 = 0.15
        pos_ee_local = np.array([0.01, 0.0, 0.0])
        pos_ee = pos_link4 + np.dot(rot_link4, pos_ee_local)
        
        # Apply base rotation and translation
        ee_pos_world = np.dot(base_rotation_matrix, pos_ee) + base_pos
        ee_rot_world = np.dot(base_rotation_matrix, rot_link4)
        
        # Convert rotation matrix to quaternion
        ee_quat = quaternion_from_rotation_matrix(ee_rot_world)
        
        return np.concatenate((ee_pos_world, ee_quat))
    
    # generate_dh_matrix method removed since we're using direct transformations

    def _convert_x_opt_to_ee_poses(self, x_opt):
        """Convert MPC state trajectory to EE poses using forward kinematics.
        
        Args:
            x_opt: MPC state trajectory of shape (N+1, 16)
                   [p(3), v(3), base_euler(3), base_ang_vel(3), arm_angle(4)]
        
        Returns:
            ee_poses: Array of shape (N+1, 7) containing [pos(3), quat_wxyz(4)]
        """
        ee_poses = []
        
        for i in range(x_opt.shape[0]):
            state = x_opt[i]
            base_pos = state[0:3]
            base_euler = state[6:9]
            arm_angle = state[12:16]
            
            # Use forward kinematics to get EE pose
            ee_pose = self.forward_kinematics(base_pos, base_euler, arm_angle)
            ee_poses.append(ee_pose)
        
        return np.array(ee_poses)

    def reset(self, reset_qp_solver_mem=1):
        """Reset the MPC planner and clear internal state"""
        # Reset the underlying MPC solver
        self.mpc.reset(reset_qp_solver_mem)
        
        # Reset internal state variables
        self.x0 = np.zeros(16)  # State: p, v, base_euler, base_ang_vel, arm_angle
        self.u_cmd_last = None  # Clear filtered command history