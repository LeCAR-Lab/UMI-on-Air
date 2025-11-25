import casadi as ca
import numpy as np

from planner.ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, quaternion_to_rpy, rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca

import matplotlib.pyplot as plt

EPS = 1e-4
IDX_BASE_X = 0 # thrust x
IDX_BASE_Y = 1 # thrust y
IDX_BASE_Z = 2 # thrust z
IDX_BASE_YAW = 3 # atti z
IDX_JOINT1 = 4 # joint1
IDX_JOINT2 = 5 # joint2

# drone state status
TAKEOFF = 1
PAUSE = 2
LAND = 3
FREEFLIGHT = 4


HORIZON = 1.0  # 0.1s
dt = 0.05  # 50Hz
MPC_STEPS = int(HORIZON/dt)  # 10 steps
print("MPC_STEPS: ", MPC_STEPS)
DEBUG = True


# during takeoff, pause, land, the arm is not moving


def init_mpc_DH():
    ik = DHMPCPlanner()
    return ik



class DHMPCPlanner:
    """
    Solved the IK using forward kinematics DH parameters
    """

    def __init__(self):

        # alpha0, a0, d0, theta_0,


        self.DH_params = np.array([[-np.pi/2, 0, 0, 0],
                                [-np.pi/2, 0.364, 0.0, -3.14],
                                [-np.pi/2, 0.0, 0.084, 0],
                                [-np.pi/2, 0.473, 0.0, -3.14]])
        
        # Initial Optimization
        self.opti_init()
        


    def init_takeoff_arm_joints(self):
        self.takeoff_arm_joints = np.array([0.0, 0.0])

        
    def opti_init(self):
        # Create an optimization problem
        self.opti = ca.Opti()
        n_state = 8
        # Optimization
        self.thrust_var = self.opti.variable(MPC_STEPS, 3)
        self.yaw_torque_var = self.opti.variable(MPC_STEPS)
        self.joint_angle_delta_var = self.opti.variable(MPC_STEPS, 2)
        
        

        self.init_takeoff_arm_joints()


        # Define the parameters
        self.jnt1_lb = self.opti.parameter()
        self.jnt1_ub = self.opti.parameter()
        self.jnt2_lb = self.opti.parameter()
        self.jnt2_ub = self.opti.parameter()
        self.base_x_ub = self.opti.parameter()
        self.opti.set_value(self.base_x_ub, 0.3)
        
        self.ee_pos_tar = self.opti.parameter(3)
        self.ee_quat_tar = self.opti.parameter(4)
        
        self.cur_joints = self.opti.parameter(2)
        self.cur_base_pos = self.opti.parameter(3)
        self.cur_base_vel = self.opti.parameter(3)
        self.cur_yaw_angle = self.opti.parameter()
        self.cur_yaw_rate = self.opti.parameter()
        
        obj = self.objective(MPC_STEPS, self.thrust_var, self.yaw_torque_var, self.joint_angle_delta_var, self.cur_base_pos, self.cur_base_vel, self.cur_yaw_angle, self.cur_yaw_rate, self.cur_joints, self.ee_pos_tar, self.ee_quat_tar)
        
        # print("obj: ", obj)
        
        self.opti.minimize(obj)
        
        
        # Set solver options
        self.opts = {
                    "verbose": False,             # CasADi verbosity, False means less output
                    "print_time": True,          # Whether to print the optimization time
                    "ipopt": {
                        "print_level": 0,         # IPOPT print level, 0 is silent
                        "max_iter": 100,          # Maximum number of iterations
                    }
                }
        self.opti.solver("ipopt", self.opts)
        
    def base_constrain_callback(self, base_pos_ub, base_pos_lb):
        self.opti.set_value(self.base_x_ub, base_pos_ub[0])


    def generate_dh_matrix(self, alpha, a, d, theta):

        transform_matrix = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                            [0, np.sin(alpha), np.cos(alpha), d],
                            [0, 0, 0, 1]])

        return transform_matrix

    def generate_dh_matrix_ca(self, alpha, a, d, theta):
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
    
    
    def objective(self, horizon, thrust, yaw_torque, joint_angle_delta, cur_base_pos, cur_base_vel, cur_yaw_angle, cur_yaw_rate, cur_joint_angle, ee_pos_tar, ee_quat_tar):
        base_pos = cur_base_pos
        base_vel = cur_base_vel
        yaw_angle = cur_yaw_angle
        yaw_rate = cur_yaw_rate
        joint_angle = cur_joint_angle
        
        ee_pos_error = 0
        ee_quat_error = 0
        
        base_lin_vel_cost = 0
        base_ang_vel_cost = 0
        
        manipulator_cost = 0
        manipulator_joint_change = 0
        
        thrust_cost = 0
        torque_cost = 0
        
        self.ee_pos_traj = []
        self.ee_quat_traj = []
        self.base_vel_traj = []
        self.base_yaw_rate_traj = []
        self.joint_angles_traj = []
        
        
        for i in range(horizon):
            base_pos, base_vel, yaw_angle, yaw_rate, joint_angle = self.forward_dynamics_ca(thrust[i, :], yaw_torque[i, :], joint_angle_delta[i, :], base_pos, base_vel, yaw_angle, yaw_rate, joint_angle)
            
            pred_ee_state = self.forward_kinematics_ca(base_pos, yaw_angle, joint_angle)
            pred_ee_pos = pred_ee_state[:3]
            pred_ee_quat = pred_ee_state[3:]
            
            ee_pos_error += ca.sumsqr(pred_ee_pos - ee_pos_tar)*0.1
            ee_quat_error += (1 - ca.dot(pred_ee_quat, ee_quat_tar)**2)*10.0
            
            manipulator_cost += (ca.sumsqr(joint_angle))*0.001
            manipulator_joint_change += ca.sumsqr(joint_angle_delta[i])*0.00001
            
            thrust_cost += ca.sumsqr(thrust[i])*0.00001
            torque_cost += ca.sumsqr(yaw_torque[i])*0.1
            
            base_lin_vel_cost += ca.sumsqr(base_vel)*0.01
            base_ang_vel_cost += ca.sumsqr(yaw_rate)*0.01
            
            # ctrl constraints
            self.opti.subject_to(thrust[i] >= ca.DM([-0.5, -0.5, -0.5]))
            self.opti.subject_to(thrust[i] <= ca.DM([0.5, 0.5, 0.5]))
            self.opti.subject_to(yaw_torque[i] >= -0.1)
            self.opti.subject_to(yaw_torque[i] <= 0.1)
            self.opti.subject_to(joint_angle_delta[i] >= ca.DM([-0.1, -0.1]))
            self.opti.subject_to(joint_angle_delta[i] <= ca.DM([0.1, 0.1]))
            
            # state constraints
            if i != 0:
                self.opti.subject_to(base_pos[0] <= self.base_x_ub)
                self.opti.subject_to(yaw_angle >= -np.pi)
                self.opti.subject_to(yaw_angle <= np.pi)
                
            self.opti.subject_to(joint_angle[0] >= self.jnt1_lb)
            self.opti.subject_to(joint_angle[0] <= self.jnt1_ub)
            self.opti.subject_to(joint_angle[1] >= self.jnt2_lb)
            self.opti.subject_to(joint_angle[1] <= self.jnt2_ub)
            
            
            if DEBUG:
                # logging
                self.ee_pos_traj.append(pred_ee_pos)
                self.ee_quat_traj.append(pred_ee_quat)
                self.base_vel_traj.append(base_vel)
                self.base_yaw_rate_traj.append(yaw_rate)
                self.joint_angles_traj.append(joint_angle)
        
        if DEBUG:
            self.thrust = thrust
            self.yaw_torque = yaw_torque
            self.joint_angle_delta = joint_angle_delta
        
        
        # terminal cost
        ter_pose_error = ca.sumsqr(pred_ee_pos - ee_pos_tar) * 100
        ter_quat_error = (1 - ca.dot(pred_ee_quat, ee_quat_tar)**2)*0.1
        ter_base_lin_vel_err = ca.sumsqr(base_vel)
        ter_base_ang_vel_err = ca.sumsqr(yaw_rate)
        
        self.predict_ee_state = pred_ee_state
        
            
        obj = ee_pos_error + ee_quat_error + manipulator_cost + manipulator_joint_change + thrust_cost + torque_cost + base_lin_vel_cost + base_ang_vel_cost
        
        obj += (ter_pose_error + ter_quat_error + ter_base_lin_vel_err + ter_base_ang_vel_err) * 10.0
        
        
        return obj
        
        
    
    
    def forward_dynamics_ca(self, thrust, yaw_torque, joint_angle_delta, base_pos, base_vel, yaw_angle, yaw_rate, joint_angle):
        # print("thrust: ", thrust.shape)
        thrust2acc_gain = ca.DM([3.0, 3.0, 6.0])
        new_base_vel = base_vel + thrust2acc_gain * thrust.T * dt
        new_base_pos = base_pos + base_vel * dt
        new_yaw_rate = yaw_rate + yaw_torque * dt * 1e-9
        new_yaw_angle = yaw_angle + yaw_rate * dt
        new_joint_angle = joint_angle + joint_angle_delta.T
        return new_base_pos, new_base_vel, new_yaw_angle, new_yaw_rate, new_joint_angle
        
    
    
    
    def forward_kinematics_ca(self, base_pos, base_yaw, joint_angles):
        # u = [x, y, z, theta1, theta2]
        
        # base_pos = u[0:3]
        # # w, x, y, z
        # base_ori = u[3:6]  # Default quaternion in CasADi
        full_joint_angles = ca.vertcat(0.0, joint_angles[0], 0.0, joint_angles[1])  # Correctly construct a CasADi vector
        base_ori = ca.vertcat(0.0, 0.0, base_yaw)

        T = ca.MX.eye(4)  # Identity matrix in CasADi

        for i in range(4):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + full_joint_angles[i]
            # T = np.dot(T, self.generate_dh_matrix(alpha, a, d, theta))
            T = ca.mtimes(T, self.generate_dh_matrix_ca(alpha, a, d, theta))


        ee_pos = ca.MX([0, 0, 0, 1])
        ee_pos = ca.mtimes(T, ee_pos)
        ee_pos = ee_pos[:3]/ee_pos[3]
        ee_orientation_matrix = T[:3, :3]
        

        base_rotation_matrix = rotation_matrix_from_euler_ca(base_ori)

        ee_pos = ca.mtimes(base_rotation_matrix, ee_pos[:3])
        ee_pos = ee_pos + base_pos
        ee_orientation_matrix = ca.mtimes(base_rotation_matrix, ee_orientation_matrix)
        ee_orientation = quaternion_from_rotation_matrix_ca(ee_orientation_matrix)
        ee_state = ca.vertcat(ee_pos, ee_orientation)

        return ee_state
    


    def optimize(self, current_drone_state, target_ee_pos, target_ee_quat, drone_state_status):
        '''
        input:
        current_drone_state: [pos_x, pos_y, pos_z, w, x, y, z, theta1, theta2]
        target_ee_quat: [w, x, y, z]
        '''


        current_base_pos = current_drone_state[:3]
        # w, x, y, z
        current_base_quat = current_drone_state[3:7]
        current_base_ori = quaternion_to_rpy(current_base_quat)
        # print("current_base_quat: ", current_base_quat)
        # print("current_base_ori: ", current_base_ori*180.0/np.pi)


        current_manipulator_joints = current_drone_state[7:9]

        target_ee_state = np.concatenate((target_ee_pos, target_ee_quat))
        
        self.target_ee_pos = target_ee_pos
        self.target_ee_ori = quaternion_to_rpy(target_ee_quat)
        

        # init_guess = np.array([current_base_pos[0], current_base_pos[1], current_base_pos[2],
        #                         current_base_ori[0], current_base_ori[1], current_base_ori[2],
        #                       current_manipulator_joints[0], current_manipulator_joints[1]])
        self.opti.set_initial(self.thrust_var, 0.0)
        self.opti.set_initial(self.yaw_torque_var, 0.0)
        self.opti.set_initial(self.joint_angle_delta_var, 0.0)

        
        self.opti.set_value(self.ee_pos_tar, target_ee_pos)
        self.opti.set_value(self.ee_quat_tar, target_ee_quat)
        self.opti.set_value(self.cur_joints, current_manipulator_joints)
        self.opti.set_value(self.cur_base_pos, current_base_pos)
        self.opti.set_value(self.cur_base_vel, np.array([0.0, 0.0, 0.0]))
        self.opti.set_value(self.cur_yaw_angle, current_base_ori[2])
        self.opti.set_value(self.cur_yaw_rate, 0.0)
        
        
        if drone_state_status == TAKEOFF:
            # the arm is not moving
            # the abs between the arm joint diff is less than eps
            self.opti.set_value(self.jnt1_lb, self.takeoff_arm_joints[0] - EPS)
            self.opti.set_value(self.jnt1_ub, self.takeoff_arm_joints[0] + EPS)
            self.opti.set_value(self.jnt2_lb, self.takeoff_arm_joints[1] - EPS)
            self.opti.set_value(self.jnt2_ub, self.takeoff_arm_joints[1] + EPS)
        else:
            # print("FREEFLIGHT")
            self.opti.set_value(self.jnt1_lb, -0.1)
            self.opti.set_value(self.jnt1_ub, 170/180*np.pi)
            self.opti.set_value(self.jnt2_lb, -0.1)
            self.opti.set_value(self.jnt2_ub, 170/180*np.pi)
            

        sol = self.opti.solve()
        
        # Get the optimized joint angles
        
        thrust = sol.value(self.thrust_var)
        yaw_torque = sol.value(self.yaw_torque_var)
        joint_angle_delta = sol.value(self.joint_angle_delta_var)
        
        obj_val = sol.value(self.opti.f)
        # print("obj_val: ", obj_val)
        

        # verify the solution
        predict_ee_state = sol.value(self.predict_ee_state)
        predict_ee_state = np.array(predict_ee_state)

        predict_ee_pos = predict_ee_state[:3]
        predict_ee_orientation = predict_ee_state[3:]
        pose_error = np.sum((predict_ee_pos - target_ee_pos)**2)
        # print("current_drone_state: ", drone_state_status)
        # print("current_base_pos: ", current_base_pos)
        # print("target_ee_pos: ", target_ee_pos)
        # print("predict_ee_pos: ", predict_ee_pos)
        # print("target_ee_quat: ", target_ee_quat)
        # print("predict_ee_orientation: ", predict_ee_orientation)
        # print("u: ", u)

        # print("pose_error: ", pose_error)
        # if pose_error > 0.01:
        #     print(f"Warning: pose error is too large: {pose_error}")
            
        # print("thrust: ", thrust)
        # print("yaw_torque: ", yaw_torque)
        # print("joint_angle_delta: ", joint_angle_delta)    
        

        return thrust, yaw_torque, joint_angle_delta

        # input("Press Enter to continue...")
        # Yuanhang: rpy to quat
        # km_target_quat = quaternion_from_rotation_matrix(rpy_to_rotation_matrix(km_target[3:6]))
        # km_target = np.array([km_target[0], km_target[1], km_target[2], 
        #                       km_target_quat[0], km_target_quat[1], km_target_quat[2], km_target_quat[3], 
        #                       km_target[6], km_target[7]])
        # return km_target
    
    def plot_last_result(self):
        pred_ee_pos = []
        pred_ee_ori = []
        pred_base_vel = []
        pred_base_yaw_rate = []
        pred_joint_angles = []
        thrust = self.opti.value(self.thrust)
        yaw_torque = self.opti.value(self.yaw_torque)
        joint_angle_delta = self.opti.value(self.joint_angle_delta)
        
        for i in range(len(self.ee_pos_traj)):
            pred_ee_pos.append(self.opti.value(self.ee_pos_traj[i]))
            ee_quat = self.opti.value(self.ee_quat_traj[i])
            pred_ee_ori.append(quaternion_to_rpy(ee_quat))
            pred_base_vel.append(self.opti.value(self.base_vel_traj[i]))
            pred_base_yaw_rate.append(self.opti.value(self.base_yaw_rate_traj[i]))
            pred_joint_angles.append(self.opti.value(self.joint_angles_traj[i]))
            
            
        pred_ee_pos = np.array(pred_ee_pos)
        pred_ee_ori = np.array(pred_ee_ori)
        pred_base_vel = np.array(pred_base_vel)
        pred_base_yaw_rate = np.array(pred_base_yaw_rate)
        pred_joint_angles = np.array(pred_joint_angles)
        
        fig, ax = plt.subplots(3, 5, figsize=(20, 10))
        
        ax[0, 0].plot(pred_ee_pos[:, 0], label="Predicted EE x")
        ax[0, 0].plot([self.target_ee_pos[0]]*len(pred_ee_pos), '--', label="Target EE x")
        ax[0, 0].set_title("Predicted EE x")
        ax[0, 0].legend()
        
        ax[1, 0].plot(pred_ee_pos[:, 1], label="Predicted EE y")
        ax[1, 0].plot([self.target_ee_pos[1]]*len(pred_ee_pos), '--', label="Target EE y")
        ax[1, 0].set_title("Predicted EE y")
        ax[1, 0].legend()
        
        ax[2, 0].plot(pred_ee_pos[:, 2], label="Predicted EE z")
        ax[2, 0].plot([self.target_ee_pos[2]]*len(pred_ee_pos), '--', label="Target EE z")
        ax[2, 0].set_title("Predicted EE z")
        ax[2, 0].legend()
        
        ax[0, 1].plot(pred_ee_ori[:, 2], label="Predicted EE yaw")
        ax[0, 1].plot([self.target_ee_ori[2]]*len(pred_ee_ori), '--', label="Target EE yaw")
        ax[0, 1].set_title("Predicted EE yaw")
        ax[0, 1].legend()
        
        ax[1, 1].plot(pred_base_yaw_rate, label="Predicted base yaw rate")
        ax[1, 1].set_title("Predicted base yaw rate")
        ax[1, 1].legend()
        
        ax[2, 1].plot(yaw_torque, label="Predicted yaw torque")
        ax[2, 1].set_title("Predicted yaw torque")
        ax[2, 1].legend()
        
        ax[0, 2].plot(pred_base_vel[:, 0], label="Predicted base vel x")
        ax[0, 2].set_title("Predicted base vel x")
        ax[0, 2].legend()
        
        ax[1, 2].plot(pred_base_vel[:, 1], label="Predicted base vel y")
        ax[1, 2].set_title("Predicted base vel y")
        ax[1, 2].legend()
        
        ax[2, 2].plot(pred_base_vel[:, 2], label="Predicted base vel z")
        ax[2, 2].set_title("Predicted base vel z")
        ax[2, 2].legend()
        
        ax[0, 3].plot(thrust[:, 0], label="Predicted thrust x")
        ax[0, 3].set_title("Predicted thrust x")
        ax[0, 3].legend()
        
        ax[1, 3].plot(thrust[:, 1], label="Predicted thrust y")
        ax[1, 3].set_title("Predicted thrust y")
        ax[1, 3].legend()

        ax[2, 3].plot(thrust[:, 2], label="Predicted thrust z")
        ax[2, 3].set_title("Predicted thrust z")
        ax[2, 3].legend()
        
        ax[0, 4].plot(pred_joint_angles[:, 0], label="Predicted joint1")
        ax[0, 4].set_title("Predicted joint1")
        ax[0, 4].legend()
        
        ax[1, 4].plot(pred_joint_angles[:, 1], label="Predicted joint2")
        ax[1, 4].set_title("Predicted joint2")
        ax[1, 4].legend()
        
        fig.tight_layout()
        plt.show()
        
        