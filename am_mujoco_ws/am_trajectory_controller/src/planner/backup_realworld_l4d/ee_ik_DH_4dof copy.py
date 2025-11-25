import casadi as ca
import numpy as np

from planner.ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca
from scipy.spatial.transform import Rotation as R



EPS = 1e-4
IDX_BASE_X = 0
IDX_BASE_Y = 1
IDX_BASE_Z = 2
IDX_BASE_ROLL = 3
IDX_BASE_PITCH = 4
IDX_BASE_YAW = 5
IDX_JOINT1 = 6
IDX_JOINT2 = 7
IDX_JOINT3 = 8
IDX_JOINT4 = 9

# drone state status
WAITING = 0
TAKEOFF = 1
PAUSE = 2
LAND = 3
FREEFLIGHT = 4


def quaternion_to_rpy(quat, xyzw = False, degrees = False):
    if xyzw:
        quat_xyzw = quat
    if not xyzw:
        qw, qx, qy, qz = quat
        quat_xyzw = [qx, qy, qz, qw]
    r = R.from_quat(quat_xyzw)
    euler = r.as_euler('zyx', degrees=degrees)
    euler = euler[::-1]
    return euler

def rpy_to_quaternion(rpy, xyzw = False, degrees = False):
    ypr = rpy[::-1]
    r = R.from_euler('zyx', ypr, degrees=False)
    quat = r.as_quat()
    qx, qy, qz, qw = quat
    if xyzw:
        quat_xyzw = [qx, qy, qz, qw]
        return quat_xyzw
    else:
        quat_wxyz = [qw, qx, qy, qz]
        return quat_wxyz



# during takeoff, pause, land, the arm is not moving


def init_ik_DH(base_x_ub = 0.1):
    ik = DHIKPlanner(base_x_ub)
    return ik



class DHIKPlanner:
    """
    Solved the IK using forward kinematics DH parameters
    """

    def __init__(self, base_x_ub = 1.1):

        
        # parameters = 
        # alpha0, a0, d0, theta_0,
        # alpha1, a1, d1, theta_1,
        # alpha2, a2, d2, theta_2,
        # alpha3, a3, d3, theta_3,
        # alpha4, a4, d4, theta_4,
        # alpha5, a5, d5, theta_5,
        # x_0, y_0, z_0,
        # x_2, y_2, z_2


        
        # params = np.array([ -np.pi/2, 0.0, 0.0, 0.0658,
        #                     -np.pi/2, 0.362, 0, -np.pi,
        #                     -np.pi/2, 0.0, 0.095, -0.02,
        #                     np.pi/2, 0.4489, 0.0, -np.pi,
        #                     -np.pi/2, 0.0, 0.01, 0.0,
        #                     np.pi/2, 0.0, 0.0, -np.pi/2,
        #                     0.07, 0, -0.01,
        #                     0.05, 0, 0.02])   
        

        # params = np.array([ -np.pi/2, 0.0, 0.0, 0.00468700873,
        #                     -np.pi/2, 0.351777273, 0.0000436358684, -np.pi -0.09987229,
        #                     -np.pi/2, -0.00117346897, 0.128708534, -0.00456060744,
        #                     -np.pi/2, 0.439586948, -0.00102854631, -np.pi + 0.09987236,
        #                     np.pi/2, -0.00203322517, -0.0952567816, 0.0999923579,
        #                     -np.pi/2, -0.000542510559, -0.000268711900, -np.pi/2 - 0.00672327,
        #                     0.0746119517, -0.000115239730, -0.0107932651,
        #                     0.0511498584, -0.0201307626, -0.0296525679])


        # self.DH_params = np.array([[-np.pi/2, 0, 0, 0],
        #                         [-np.pi/2, 0.364, 0.0, -3.14],
        #                         [-np.pi/2, 0.0, 0.084, 0],
        #                         [-np.pi/2, 0.473, 0.0, -3.14],
        #                         ])
        
        # self.DH_params = np.array([[-np.pi/2, 0, 0, 0.00468700873],
        #                             [-np.pi/2, 0.351777273, 0.0000436358684, -np.pi -0.09987229],
        #                             [-np.pi/2, -0.00117346897, 0.128708534, -0.00456060744],
        #                             [-np.pi/2, 0.439586948, -0.00102854631, -np.pi + 0.09987236],
        #                             [np.pi/2, -0.00203322517, -0.0952567816, 0.0999923579],
        #                             [-np.pi/2, -0.000542510559, -0.000268711900, -np.pi/2 - 0.00672327]])

        # self.DH_params = np.array([[-np.pi/2, 0, 0, 0.0],
        #                             [-np.pi/2, 0.351777273, 0.0000436358684, -np.pi],
        #                             [-np.pi/2, -0.00117346897, 0.128708534, 0.0],
        #                             [-np.pi/2, 0.439586948, -0.00102854631, -np.pi],
        #                             [np.pi/2, -0.00203322517, -0.0952567816, 0.0],
        #                             [-np.pi/2, -0.000542510559, -0.000268711900, -np.pi/2]])
        
        self.DH_params = np.array([[-1.67079619, 0, 0, 0.0453440932],
                            [-1.49196834, 0.362895306, 0.0, -3.04159272],
                            [-1.67079052, 0.00711424939, 0.0496796518, 0.0137621784],
                            [-1.47080046, 0.441054359, 0.0, -3.24159258],
                            [1.57777222, 0.00980744858, 0.0762684723, -0.0193069740],
                            [-1.47079649, 0.0, 0.0, -1.67079626]])
                       


        # self.arm_base_pos = np.array([0.0746119517, -0.000115239730, -0.0107932651])
        # self.ee_base_pos = np.array([0.0511498584, -0.0201307626, -0.0296525679])

        self.arm_base_pos = np.array([0.0664397079, 0.0, -0.0171154472])
        self.ee_base_pos = np.array([0.149832002, -0.0189594673, -0.00617417526])

        
        # Initial Optimization
        self.opti_init(base_x_ub)
        


    def init_takeoff_arm_joints(self):
        self.takeoff_arm_joints = np.array([10.0/180.0*np.pi, 12.0/180.0*np.pi, np.pi/2, 0.0])     # in radian

        
    def opti_init(self, base_x_ub):
        # Create an optimization problem
        self.opti = ca.Opti()
        n_state = 9
        # Optimization
        self.u = self.opti.variable(n_state)

        self.init_takeoff_arm_joints()


        # Define the parameters
        self.p6_lb = self.opti.parameter()
        self.p6_ub = self.opti.parameter()
        self.p7_lb = self.opti.parameter()
        self.p7_ub = self.opti.parameter()
        self.p8_lb = self.opti.parameter()
        self.p8_ub = self.opti.parameter()

        self.current_base_x = self.opti.parameter()
        self.current_base_y = self.opti.parameter()
        self.current_base_z = self.opti.parameter()

        self.base_x_ub = self.opti.parameter()
        self.opti.set_value(self.base_x_ub, base_x_ub)
        
        self.obj_ee_pos_tar = self.opti.parameter(3)
        self.obj_ee_quat_tar = self.opti.parameter(4)
        
        self.obj_cur_joints = self.opti.parameter(3)
        
        # Define the constraints
        # zero tilt
        self.opti.subject_to(self.u[IDX_BASE_ROLL] > -EPS)
        self.opti.subject_to(self.u[IDX_BASE_ROLL] <= EPS)
        self.opti.subject_to(self.u[IDX_BASE_PITCH] > -EPS)
        self.opti.subject_to(self.u[IDX_BASE_PITCH] <= EPS)
        self.opti.subject_to(self.u[IDX_BASE_YAW] >  -30.0/180 * np.pi)
        self.opti.subject_to(self.u[IDX_BASE_YAW] <= 30.0/180 * np.pi)
        # arm joint limits
        self.opti.subject_to(self.u[IDX_JOINT1] >= self.p6_lb)
        self.opti.subject_to(self.u[IDX_JOINT1] <= self.p6_ub)
        self.opti.subject_to(self.u[IDX_JOINT2] >= self.p7_lb)
        self.opti.subject_to(self.u[IDX_JOINT2] <= self.p7_ub)
        self.opti.subject_to(self.u[IDX_JOINT3] >= self.p8_lb)
        self.opti.subject_to(self.u[IDX_JOINT3] <= self.p8_ub)


        # base limits
        # self.opti.subject_to(self.u[2] <=0.4)
        self.opti.subject_to(self.u[IDX_BASE_X] <= self.base_x_ub)
        self.opti.subject_to(self.u[IDX_BASE_Z] <= 1.5)
        
        # base change limits
        # self.opti.subject_to(self.u[IDX_BASE_X] >= self.current_base_x - 0.1)
        # self.opti.subject_to(self.u[IDX_BASE_X] <= self.current_base_x + 0.1)
        # self.opti.subject_to(self.u[IDX_BASE_Y] >= self.current_base_y - 0.1)
        # self.opti.subject_to(self.u[IDX_BASE_Y] <= self.current_base_y + 0.1)
        # self.opti.subject_to(self.u[IDX_BASE_Z] >= self.current_base_z - 0.1)
        # self.opti.subject_to(self.u[IDX_BASE_Z] <= self.current_base_z + 0.1)

        # Define the objective
        predict_ee_state = self.forward_kinematics_ca(self.u)
        predict_ee_pos = predict_ee_state[:3]
        predict_ee_quat = predict_ee_state[3:]
        
        pose_error = ca.sumsqr(predict_ee_pos - self.obj_ee_pos_tar)*10
        orientation_error = (1 - ca.dot(predict_ee_quat, self.obj_ee_quat_tar)**2)*1.0
        
        manipulator_cost = ((self.u[IDX_JOINT1]-10.0/180.0*np.pi)**2 + (self.u[IDX_JOINT2] - 12.0/180.0*np.pi)**2 ) * 0.01
        manipulator_joint_change = ca.sumsqr(self.u[IDX_JOINT1:IDX_JOINT3+1] - self.obj_cur_joints) *0.01
        base_change = ((self.u[IDX_BASE_X] - self.current_base_x) **2 + (self.u[IDX_BASE_Y] - self.current_base_y) **2 + (self.u[IDX_BASE_Z] - self.current_base_z) **2) * 0.001
        
        obj = pose_error + orientation_error + manipulator_cost + manipulator_joint_change + base_change
        # obj = pose_error + orientation_error

        self.opti.minimize(obj)
        
        
        # Set solver options
        self.opts = {
                    "verbose": False,             # CasADi verbosity, False means less output
                    "print_time": False,          # Whether to print the optimization time
                    "ipopt": {
                        "print_level": 0,         # IPOPT print level, 0 is silent
                        "max_iter": 1000,          # Maximum number of iterations
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
        '''
            input: u: [base_pos, base_euler(rpy), joint_angles]
        '''

        base_pos = u[0:3]
        # base roll, pitch, yaw
        base_euler = u[3:6]

        joint_angles = ca.vertcat(0.0, u[6], 0.0, u[7], 0.0, u[8])      # Correctly construct a CasADi vector

        arm_base_pos = ca.MX([self.arm_base_pos[0], self.arm_base_pos[1], self.arm_base_pos[2]])

        T = ca.MX.eye(4)  # Identity matrix in CasADi

        for i in range(6):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
            # T = np.dot(T, self.generate_dh_matrix(alpha, a, d, theta))
            T = ca.mtimes(T, self.generate_dh_matrix_ca(alpha, a, d, theta))

        # ee_pos = ca.MX([0, 0, 0, 1])
        ee_pos = ca.MX([self.ee_base_pos[0], self.ee_base_pos[1], self.ee_base_pos[2], 1])
        ee_pos = ca.mtimes(T, ee_pos)
        ee_pos = ee_pos[:3]/ee_pos[3]
        ee_orientation_matrix = T[:3, :3]
        
        base_rotation_matrix = rotation_matrix_from_euler_ca(base_euler)

        ee_pos = ee_pos + arm_base_pos
        ee_pos = ca.mtimes(base_rotation_matrix, ee_pos[:3])
        ee_pos = ee_pos + base_pos
        ee_orientation_matrix = ca.mtimes(base_rotation_matrix, ee_orientation_matrix)
        ee_quat = quaternion_from_rotation_matrix_ca(ee_orientation_matrix)
        ee_state = ca.vertcat(ee_pos, ee_quat)

        return ee_state
    
    def forward_kinematics(self, u, base_rpy_use = True):
        '''
            input: u: [base_pos, base_euler(rpy), joint_angles]
            output: ee_state: [ee_pos, ee_quat]
        '''

        base_pos = u[0:3]

        if base_rpy_use:
            # base roll, pitch, yaw
            base_euler = u[3:6]
            joint_angles = np.array([0.0, u[6], 0.0, u[7], 0.0,  u[8]])
        else:
            # base quat
            base_quat = u[3:7]
            base_euler = quaternion_to_rpy(base_quat, xyzw = False, degrees = False)
            joint_angles = np.array([0.0, u[7], 0.0, u[8], 0.0,  u[9]])


        T = np.eye(4)

        for i in range(6):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = theta + joint_angles[i]
            dh_matrix = self.generate_dh_matrix(alpha, a, d, theta)
            T = np.dot(T, dh_matrix)

        ee_pos = np.array([self.ee_base_pos[0], self.ee_base_pos[1], self.ee_base_pos[2], 1])
        ee_pos = np.dot(T, ee_pos)
        ee_pos = ee_pos[:3]/ee_pos[3]
        ee_orientation_matrix = T[:3, :3]
        

        base_rotation_matrix = rpy_to_rotation_matrix(base_euler)
        ee_pos = ee_pos + self.arm_base_pos
        ee_pos = np.dot(base_rotation_matrix, ee_pos[:3])
        ee_pos = ee_pos + base_pos
        ee_orientation_matrix = np.dot(base_rotation_matrix, ee_orientation_matrix)
        ee_quat = quaternion_from_rotation_matrix(ee_orientation_matrix)
        ee_state = np.concatenate((ee_pos, ee_quat))

        return ee_state


    def optimize(self, current_drone_state, target_ee_pos, target_ee_quat, drone_state_status):
        '''
        input:
        current_drone_state: [pos_x, pos_y, pos_z, w, x, y, z, theta1, theta2, theta_3]
        target_ee_quat: [w, x, y, z]

        output:
        km_target: [pos_x, pos_y, pos_z, w, x, y, z, theta1, theta2, theta3]
        '''


        current_base_pos = current_drone_state[:3]
        # w, x, y, z
        current_base_quat = current_drone_state[3:7]
        current_base_euler = quaternion_to_rpy(current_base_quat, xyzw = False, degrees = False)
        current_manipulator_joints = current_drone_state[7:10]
        
        init_guess = np.array([current_base_pos[0], current_base_pos[1], current_base_pos[2],
                                current_base_euler[0], current_base_euler[1], current_base_euler[2],
                              current_manipulator_joints[0], current_manipulator_joints[1], current_manipulator_joints[2]])
        # predict_ee_state = self.forward_kinematics(init_guess)
        # predict_ee_pos = predict_ee_state[:3]
        # predict_ee_quat = predict_ee_state[3:7]
        # print("predict_ee_pos from current: ", predict_ee_pos)
        # print("predict_ee_quat from current: ", predict_ee_quat)


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
            self.opti.set_value(self.p8_lb, self.takeoff_arm_joints[2] - EPS)
            self.opti.set_value(self.p8_ub, self.takeoff_arm_joints[2] + EPS)

            self.opti.set_value(self.current_base_x, current_base_pos[0])
            self.opti.set_value(self.current_base_y, current_base_pos[1])
            self.opti.set_value(self.current_base_z, current_base_pos[2])


        else:
            # print("FREEFLIGHT")
            self.opti.set_value(self.p6_lb, 8/180*np.pi)
            self.opti.set_value(self.p6_ub, 130/180*np.pi)
            self.opti.set_value(self.p7_lb, 10/180*np.pi)
            self.opti.set_value(self.p7_ub, 130/180*np.pi)
            self.opti.set_value(self.p8_lb, 30/180*np.pi)
            self.opti.set_value(self.p8_ub, 150/180*np.pi)


            self.opti.set_value(self.current_base_x, current_base_pos[0])
            self.opti.set_value(self.current_base_y, current_base_pos[1])
            self.opti.set_value(self.current_base_z, current_base_pos[2])

        sol = self.opti.solve()
        
        # Get the optimized joint angles
        km_target = sol.value(self.u)

        # if not km_target:
        #     print("Warning: Optimization failed")
        #     km_target = init_guess

        # verify the solution
        km_target = np.array([km_target[0], km_target[1], km_target[2], km_target[3], km_target[4], km_target[5], km_target[6], km_target[7], km_target[8]])
        
        predict_ee_state = self.forward_kinematics(km_target)

        predict_ee_pos = predict_ee_state[:3]
        predict_ee_orientation = predict_ee_state[3:]
        pose_error = np.sum((predict_ee_pos - target_ee_pos)**2)
        orientation_error = 1 - np.dot(predict_ee_orientation, target_ee_quat)**2
        # print("current_drone_state: ", drone_state_status)
        # print("current_base_pos: ", current_base_pos)
        # print("km_target: ", km_target)
        # print("target_ee_pos: ", target_ee_pos)
        # print("predict_ee_pos: ", predict_ee_pos)
        # print("target_ee_quat: ", target_ee_quat)
        # print("predict_ee_orientation: ", predict_ee_orientation)
        # print("u: ", u)

        # print("pose_error: ", pose_error)
        # print("orientation_error: ", orientation_error)
        if pose_error > 0.01:
            print(f"Warning: pose error is too large: {pose_error}")

        # input("Press Enter to continue...")

        km_target_quat = rpy_to_quaternion(km_target[3:6], xyzw = False, degrees = False)

        km_target = np.array([km_target[0], km_target[1], km_target[2], 
                              km_target_quat[0], km_target_quat[1], km_target_quat[2], km_target_quat[3], 
                              km_target[6], km_target[7], km_target[8]])
        return km_target
    
    