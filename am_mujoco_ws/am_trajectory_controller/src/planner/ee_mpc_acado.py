# mpc_file.py

import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

class DroneMPC:
    def __init__(self, mass, T, N, Q, R, acc2thrust_gain, pos_min, pos_max, vel_min, vel_max, acc_min, acc_max):
        self.mass = mass
        self.T = T
        self.N = N
        self.Q = np.diag(Q)
        self.R = np.diag(R)
        self.acc2thrust_gain = np.array(acc2thrust_gain)
        self.pos_min = np.array(pos_min)
        self.pos_max = np.array(pos_max)
        self.vel_min = np.array(vel_min)
        self.vel_max = np.array(vel_max)
        self.acc_min = np.array(acc_min)
        self.acc_max = np.array(acc_max)
        self.p = cs.MX.sym('p', 3)
        self.v = cs.MX.sym('v', 3)
        self.x = cs.vertcat(self.p, self.v)
        self.fx = cs.MX.sym('fx')
        self.fy = cs.MX.sym('fy')
        self.fz = cs.MX.sym('fz')
        self.u = cs.vertcat(self.fx, self.fy, self.fz)
        self.x_dot = self.build_dynamics(self.x, self.u)
        self.model = self.build_acados_model()
        self.ocp_solver = self.build_acados_ocp_solver()

    def build_dynamics(self, x, u):
        p = x[0:3]
        v = x[3:6]
        dp = v
        dv = u / self.mass
        return cs.vertcat(dp, dv)

    def build_acados_model(self):
        x_dot = cs.MX.sym('x_dot', 6)
        f_expl = self.x_dot
        f_impl = x_dot - f_expl
        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl
        model.x = self.x
        model.xdot = x_dot
        model.u = self.u
        model.p = []
        model.name = 'fully_actuated_uav'
        return model

    def build_acados_ocp_solver(self):
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T
        nx = self.model.x.size()[0]
        nu = self.model.u.size()[0]
        ny = nx + nu
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        W = np.block([
            [self.Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), self.R]
        ])
        ocp.cost.W = W
        ocp.cost.W_e = 10.0 * self.Q
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)
        
        x_min = np.concatenate([self.pos_min, self.vel_min])
        x_max = np.concatenate([self.pos_max, self.vel_max])
        u_min = self.acc_min * self.mass
        u_max = self.acc_max * self.mass

        ocp.constraints.lbx = x_min
        ocp.constraints.ubx = x_max
        ocp.constraints.idxbx = np.array(list(range(nx)))
        ocp.constraints.lbu = u_min
        ocp.constraints.ubu = u_max
        ocp.constraints.idxbu = np.array(list(range(nu)))
        
        ocp.constraints.x0 = np.zeros(nx)
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.compile_dir = "./acados_mpc_build"
        return AcadosOcpSolver(ocp)

    def set_reference_sequence(self, p_refs, v_refs, u_ref=None):
        if u_ref is None:
            u_ref = [0, 0, 0]
        for i in range(self.N):
            x_ref = np.concatenate([p_refs[i], v_refs[i]])
            ref = np.concatenate([x_ref, u_ref])
            self.ocp_solver.set(i, "yref", ref)
        final_x_ref = np.concatenate([p_refs[-1], v_refs[-1]])
        self.ocp_solver.set(self.N, "yref", final_x_ref)

    def run_optimization(self, x0):
        self.ocp_solver.set(0, 'lbx', x0)
        self.ocp_solver.set(0, 'ubx', x0)
        self.ocp_solver.solve()
        u_opt = np.array([self.ocp_solver.get(i, "u") for i in range(self.N)])
        x_opt = np.array([self.ocp_solver.get(i, "x") for i in range(self.N + 1)])
        return u_opt, x_opt


class BaseMPCPlanner:
    def __init__(self, mass, T, N, Q, R, acc2thrust_gain, pos_min, pos_max, vel_min, vel_max, acc_min, acc_max):
        self.mpc = DroneMPC(mass, T, N, Q, R, acc2thrust_gain, pos_min, pos_max, vel_min, vel_max, acc_min, acc_max)
        self.x0 = np.zeros(6)

    def optimize(self, p, v, p_refs, v_refs, last_u):
        self.x0 = np.concatenate([p, v])
        self.mpc.set_reference_sequence(p_refs, v_refs, last_u)
        u_opt, x_opt = self.mpc.run_optimization(self.x0)
        p_opt = x_opt[:, 0:3]
        v_opt = x_opt[:, 3:6]
        u_opt = u_opt * self.mpc.acc2thrust_gain.reshape(1, 3)
        return u_opt[0], p_opt[0], v_opt[0]



import numpy as np

class DisturbanceObserver:
    def __init__(self, cutoff_freq, acc2thrust_gain, dt, acc_min, acc_max):
        # set cutoff frequency, thrust gain, and time step
        self.cutoff_freq = cutoff_freq
        self.acc2thrust_gain = acc2thrust_gain
        self.dt = dt
        self.acc_min = acc_min
        self.acc_max = acc_max
        
        # store previous velocity
        self.prev_vel = 0.0
        
        # store filtered disturbance
        self.dist_acc_filt = 0.0
        
        # precompute alpha for low-pass filter (first-order)
        self.alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_freq * self.dt)

    def update(self, current_vel, desired_acc):
        # differentiate velocity to get actual acceleration
        actual_acc = (current_vel - self.prev_vel) / self.dt
        
        # compute disturbance in acceleration domain
        dist_acc = actual_acc - desired_acc
        
        # apply low-pass filter
        self.dist_acc_filt += self.alpha * (dist_acc - self.dist_acc_filt)
        
        self.dist_acc_filt = np.clip(self.dist_acc_filt, self.acc_min, self.acc_max)
        
        # convert disturbance to thrust domain
        dist_thrust = - self.dist_acc_filt * self.acc2thrust_gain
        
        # store current velocity for next iteration
        self.prev_vel = current_vel
        
        
        return dist_thrust
