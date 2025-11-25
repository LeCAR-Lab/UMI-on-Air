import casadi as ca
import numpy as np

from planner.ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, quaternion_to_rpy, rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca
from planner.ik_util import euler_to_rotation_matrix, rotation_matrix_to_quaternion

EPS = 1e-4
IDX_BASE_X = 0
IDX_BASE_Y = 1
IDX_BASE_Z = 2
IDX_BASE_ROLL = 3
IDX_BASE_PITCH = 4
IDX_BASE_YAW = 5
IDX_JOINT1 = 6
IDX_JOINT2 = 7

# drone state status
WAITING = 0
TAKEOFF = 1
PAUSE = 2
LAND = 3
FREEFLIGHT = 4


# Base position limits
BASE_POS_UB = np.array([0.7, 2.0, 2.0])
BASE_POS_LB = np.array([-1.0, -2.0, 0.0])





# during takeoff, pause, land, the arm is not moving


def init_ik_DH():
    ik = DHIKPlanner()
    return ik



class DHIKPlanner:
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
        self.u = self.opti.variable(n_state)

        self.init_takeoff_arm_joints()


        # Define the parameters
        self.p6_lb = self.opti.parameter()
        self.p6_ub = self.opti.parameter()
        self.p7_lb = self.opti.parameter()
        self.p7_ub = self.opti.parameter()
        self.base_x_ub = self.opti.parameter()
        # TODO
        self.opti.set_value(self.base_x_ub, BASE_POS_UB[0])
        
        self.obj_ee_pos_tar = self.opti.parameter(3)
        self.obj_ee_quat_tar = self.opti.parameter(4)
        
        self.obj_cur_joints = self.opti.parameter(2)
        
        # Define the constraints
        # zero tilt
        self.opti.subject_to(self.u[IDX_BASE_ROLL] > -EPS)
        self.opti.subject_to(self.u[IDX_BASE_ROLL] <= EPS)
        self.opti.subject_to(self.u[IDX_BASE_PITCH] > -EPS)
        self.opti.subject_to(self.u[IDX_BASE_PITCH] <= EPS)
        self.opti.subject_to(self.u[IDX_BASE_YAW] > -np.pi)
        self.opti.subject_to(self.u[IDX_BASE_YAW] <= np.pi)
        # arm joint limits
        self.opti.subject_to(self.u[IDX_JOINT1] >= self.p6_lb)
        self.opti.subject_to(self.u[IDX_JOINT1] <= self.p6_ub)
        self.opti.subject_to(self.u[IDX_JOINT2] >= self.p7_lb)
        self.opti.subject_to(self.u[IDX_JOINT2] <= self.p7_ub)
        # base limits
        # self.opti.subject_to(self.u[2] <=0.4)
        self.opti.subject_to(self.u[IDX_BASE_X] <= self.base_x_ub)

        # Define the objective
        self.predict_ee_state = self.forward_kinematics_ca(self.u)
        predict_ee_pos = self.predict_ee_state[:3]
        predict_ee_orientation = self.predict_ee_state[3:]
        
        self.opti.subject_to(predict_ee_pos[2] > 0.09)


        pose_error = ca.sumsqr(predict_ee_pos - self.obj_ee_pos_tar)
        orientation_error = (1 - ca.dot(predict_ee_orientation, self.obj_ee_quat_tar)**2)*10.0
        
        manipulator_cost = (ca.sumsqr(self.u[IDX_JOINT1:IDX_JOINT2+1]))*0.01
        
        manipulator_joint_change = ca.sumsqr(self.u[IDX_JOINT1:IDX_JOINT2+1] - self.obj_cur_joints) *0.00001
        
        obj = pose_error + orientation_error + manipulator_cost + manipulator_joint_change

        self.opti.minimize(obj)
        
        
        # Set solver options
        self.opts = {
                    "verbose": False,             # CasADi verbosity, False means less output
                    "print_time": False,          # Whether to print the optimization time
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
    def forward_kinematics_ca(self, u):
        
        base_pos = u[0:3]
        base_ori = u[3:6] 
        joint_angles = ca.vertcat(0.0, u[6], 0.0, u[7])  # Correctly construct a CasADi vector

        T = ca.MX.eye(4)  # Identity matrix in CasADi

        for i in range(4):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
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
    
    
    def forward_kinematics(self, u):
        
        base_pos = u[0:3]
        base_ori = u[3:6]
        joint_angles = np.array([0.0, u[6], 0.0, u[7]])

        T = np.eye(4)  # Identity matrix in CasADi

        for i in range(4):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
            T = np.dot(T, self.generate_dh_matrix(alpha, a, d, theta))
            
        ee_pos = np.array([0, 0, 0, 1])
        ee_pos = np.dot(T, ee_pos)
        ee_pos = ee_pos[:3]/ee_pos[3]
        ee_orientation_matrix = T[:3, :3]
        
        base_rotation_matrix = euler_to_rotation_matrix(base_ori)

        ee_pos = np.dot(base_rotation_matrix, ee_pos[:3])
        ee_pos = ee_pos + base_pos
        ee_orientation_matrix = np.dot(base_rotation_matrix, ee_orientation_matrix)
        ee_orientation = rotation_matrix_to_quaternion(ee_orientation_matrix)
        ee_state = np.concatenate((ee_pos, ee_orientation))

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
        
        

        init_guess = np.array([current_base_pos[0], current_base_pos[1], current_base_pos[2],
                                current_base_ori[0], current_base_ori[1], current_base_ori[2],
                              current_manipulator_joints[0], current_manipulator_joints[1]])
        self.opti.set_initial(self.u, init_guess)

        
        self.opti.set_value(self.obj_ee_pos_tar, target_ee_pos)
        self.opti.set_value(self.obj_ee_quat_tar, target_ee_quat)
        self.opti.set_value(self.obj_cur_joints, current_manipulator_joints)
        
        if drone_state_status == TAKEOFF:
            # the arm is not moving
            # the abs between the arm joint diff is less than eps
            self.opti.set_value(self.p6_lb, self.takeoff_arm_joints[0] - EPS)
            self.opti.set_value(self.p6_ub, self.takeoff_arm_joints[0] + EPS)
            self.opti.set_value(self.p7_lb, self.takeoff_arm_joints[1] - EPS)
            self.opti.set_value(self.p7_ub, self.takeoff_arm_joints[1] + EPS)
        else:
            # print("FREEFLIGHT")
            self.opti.set_value(self.p6_lb, -0.1)
            self.opti.set_value(self.p6_ub, 170/180*np.pi)
            self.opti.set_value(self.p7_lb, -0.1)
            self.opti.set_value(self.p7_ub, 170/180*np.pi)
            

        sol = self.opti.solve()
        
        # Get the optimized joint angles
        km_target = sol.value(self.u)

        # verify the solution
        km_target = np.array([km_target[0], km_target[1], km_target[2], km_target[3], km_target[4], km_target[5], km_target[6], km_target[7]])
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
        if pose_error > 0.01:
            print(f"Warning: pose error is too large: {pose_error}")

        # input("Press Enter to continue...")
        # Yuanhang: rpy to quat
        km_target_quat = quaternion_from_rotation_matrix(rpy_to_rotation_matrix(km_target[3:6]))
        km_target = np.array([km_target[0], km_target[1], km_target[2], 
                              km_target_quat[0], km_target_quat[1], km_target_quat[2], km_target_quat[3], 
                              km_target[6], km_target[7]])
        return km_target
    
    