import casadi as ca
import numpy as np

from ik_util import rpy_to_rotation_matrix, quaternion_from_rotation_matrix, quaternion_to_rpy, rotation_matrix_from_euler_ca, quaternion_from_rotation_matrix_ca


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
TAKEOFF = 1
PAUSE = 2
LAND = 3
FREEFLIGHT = 4




# during takeoff, pause, land, the arm is not moving





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

        # DH Parameters corrected to match MuJoCo XML structure:
        # Frame 0: Base to arm_base (0.088, 0, 0) + arm_base to link1 (0, 0, 0.06475)
        # Frame 1: Joint 1 (pitch) - Y-axis rotation at link1
        # Frame 2: Link1 to link2 (-0.3795, 0, 0.059) 
        # Frame 3: Joint 2 (yaw) - Z-axis rotation at link2
        # Frame 4: Link2 to link3 (0.4475, 0, 0)
        # Frame 5: Joint 3 (yaw) - Z-axis rotation at link3  
        # Frame 6: Link3 to link4 (0.071, 0, 0) + Joint 4 (roll) - X-axis rotation
        # Frame 7: Link4 to ee (0.01, 0, 0) + ee to ee_tool (0.14, 0, 0) = (0.15, 0, 0)
        
        self.DH_params = np.array([
            [0.0, 0.088, 0.06475, 0.0],           # Frame 0: Base to link1 position
            [-np.pi/2, 0.0, 0.0, 0.0],            # Frame 1: Joint 1 (pitch) - Y-axis rotation
            [0.0, -0.3795, 0.059, -np.pi/2],      # Frame 2: Link1 to link2 offset
            [0.0, 0.0, 0.0, 0.0],                 # Frame 3: Joint 2 (yaw) - Z-axis rotation  
            [0.0, 0.4475, 0.0, 0.0],              # Frame 4: Link2 to link3 offset
            [0.0, 0.0, 0.0, 0.0],                 # Frame 5: Joint 3 (yaw) - Z-axis rotation
            [-np.pi/2, 0.071, 0.0, 0.0],          # Frame 6: Link3 to link4 + Joint 4 (roll) - X-axis
            [0.0, 0.15, 0.0, 0.0]                 # Frame 7: Link4 to end-effector (0.01 + 0.14)
        ])
        
        self.arm_base_pos = np.array([0.088, 0.0, 0.06475])  # Not used in new direct approach
        self.ee_base_pos = np.array([0.0, 0.0, 0.0])        # Not used in new direct approach

        # Initial Optimization

        self.opti_init(base_x_ub)
        


    def init_takeoff_arm_joints(self):
        self.takeoff_arm_joints = np.array([0.0, 0.0,np.pi/2, 0.0])

        
    def opti_init(self, base_x_ub):
        # Create an optimization problem
        self.opti = ca.Opti()
        n_state = 10
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
        self.p9_lb = self.opti.parameter()
        self.p9_ub = self.opti.parameter()

        self.base_x_ub = self.opti.parameter()
        self.opti.set_value(self.base_x_ub, base_x_ub)
        
        self.obj_ee_pos_tar = self.opti.parameter(3)
        self.obj_ee_quat_tar = self.opti.parameter(4)
        
        self.obj_cur_joints = self.opti.parameter(4)
        # Parameter holding the current base position so we can penalise motion of the base between optimisation steps
        self.obj_cur_base_pos = self.opti.parameter(3)
        
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
        self.opti.subject_to(self.u[IDX_JOINT3] >= self.p8_lb)
        self.opti.subject_to(self.u[IDX_JOINT3] <= self.p8_ub)
        self.opti.subject_to(self.u[IDX_JOINT4] >= self.p9_lb)
        self.opti.subject_to(self.u[IDX_JOINT4] <= self.p9_ub)      


        # base limits
        # self.opti.subject_to(self.u[2] <=0.4)
        self.opti.subject_to(self.u[IDX_BASE_X] <= self.base_x_ub)

        # Define the objective
        predict_ee_state = self.forward_kinematics_ca(self.u)
        predict_ee_pos = predict_ee_state[:3]
        predict_ee_quat = predict_ee_state[3:]
        
        pose_error = ca.sumsqr(predict_ee_pos - self.obj_ee_pos_tar)*100
        orientation_error = (1 - ca.dot(predict_ee_quat, self.obj_ee_quat_tar)**2)*20.
        # orientation_error = (1 - ca.dot(predict_ee_quat, self.obj_ee_quat_tar)**2)*1.0
        
        manipulator_cost = (
            self.u[IDX_JOINT1]**2 +
            self.u[IDX_JOINT2]**2) * 0.1

        
        manipulator_joint_change = ca.sumsqr(self.u[IDX_JOINT1:IDX_JOINT3] - self.obj_cur_joints[0:2]) * 0.01
        
        # Penalise changes in base position (encourage the solver to move the arm before moving the whole platform)
        base_change = ca.sumsqr(self.u[IDX_BASE_X:IDX_BASE_Z+1] - self.obj_cur_base_pos) * 1.0
        
        obj = pose_error + orientation_error + manipulator_cost + manipulator_joint_change + base_change
        # obj = pose_error + orientation_error

        self.opti.minimize(obj)
        
        
        # Set solver options
        self.opts = {
                    "verbose": False,             # CasADi verbosity, False means less output
                    "print_time": False,          # Whether to print the optimization time
                    "ipopt": {
                        "print_level": 0,         # IPOPT print level, 0 is silent
                        "max_iter": 1000,          # Increase maximum number of iterations
                        "tol": 1e-6,              # Convergence tolerance
                        "acceptable_tol": 1e-4,   # Acceptable convergence tolerance
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
        Direct transformation chain matching MuJoCo XML structure exactly
        input: u: [base_pos, base_euler(rpy), joint_angles]
        '''
        base_pos = u[0:3]
        base_euler = u[3:6]  # roll, pitch, yaw
        joint_angles = u[6:10]  # [joint1, joint2, joint3, joint4]

        # Base rotation matrix
        base_rotation_matrix = rotation_matrix_from_euler_ca(base_euler)
        
        # Build transformation chain exactly as in MuJoCo XML:
        # 1. base_link to arm_base_link: pos="0.088 0 0.0"
        pos_arm_base = ca.MX([0.088, 0.0, 0.0])
        
        # 2. arm_base_link to manipulation_link1_pitch_link: pos="0.0 0.0 0.06475"
        pos_link1 = pos_arm_base + ca.MX([0.0, 0.0, 0.06475])
        
        # 3. Apply joint1 rotation (Y-axis, pitch): axis="0 1 0"
        rot_joint1 = ca.MX.zeros(3, 3)
        c1, s1 = ca.cos(joint_angles[0]), ca.sin(joint_angles[0])
        rot_joint1[0, 0] = c1;  rot_joint1[0, 2] = s1
        rot_joint1[1, 1] = 1.0
        rot_joint1[2, 0] = -s1; rot_joint1[2, 2] = c1
        
        # 4. manipulation_link1 to manipulation_link2: pos="-0.3795 0.0 0.059" quat="0.707 0.707 0 0"
        # The quat="0.707 0.707 0 0" represents a 90° rotation around X-axis
        pos_link2_local = ca.MX([-0.3795, 0.0, 0.059])
        pos_link2 = pos_link1 + ca.mtimes(rot_joint1, pos_link2_local)
        
        # Link2 has initial rotation quat="0.707 0.707 0 0" = 90° around X
        rot_link2_initial = ca.MX.zeros(3, 3)
        rot_link2_initial[0, 0] = 1.0
        rot_link2_initial[1, 1] = 0.0; rot_link2_initial[1, 2] = -1.0
        rot_link2_initial[2, 1] = 1.0; rot_link2_initial[2, 2] = 0.0
        
        # 5. Apply joint2 rotation (Z-axis, yaw): axis="0 0 1"  
        rot_joint2 = ca.MX.zeros(3, 3)
        c2, s2 = ca.cos(joint_angles[1]), ca.sin(joint_angles[1])
        rot_joint2[0, 0] = c2;  rot_joint2[0, 1] = -s2
        rot_joint2[1, 0] = s2;  rot_joint2[1, 1] = c2
        rot_joint2[2, 2] = 1.0
        
        # Combined rotation for link2
        rot_link2 = ca.mtimes(ca.mtimes(rot_joint1, rot_link2_initial), rot_joint2)
        
        # 6. manipulation_link2 to manipulation_link3: pos="0.4475 0 0"
        pos_link3_local = ca.MX([0.4475, 0.0, 0.0])
        pos_link3 = pos_link2 + ca.mtimes(rot_link2, pos_link3_local)
        
        # 7. Apply joint3 rotation (Z-axis, yaw): axis="0 0 1"
        rot_joint3 = ca.MX.zeros(3, 3)
        c3, s3 = ca.cos(joint_angles[2]), ca.sin(joint_angles[2])
        rot_joint3[0, 0] = c3;  rot_joint3[0, 1] = -s3
        rot_joint3[1, 0] = s3;  rot_joint3[1, 1] = c3
        rot_joint3[2, 2] = 1.0
        
        rot_link3 = ca.mtimes(rot_link2, rot_joint3)
        
        # 8. manipulation_link3 to manipulation_link4: pos="0.071 0 0"
        pos_link4_local = ca.MX([0.071, 0.0, 0.0])
        pos_link4 = pos_link3 + ca.mtimes(rot_link3, pos_link4_local)
        
        # 9. Apply joint4 rotation (negative X-axis, roll): axis="-1 0 0"
        rot_joint4 = ca.MX.zeros(3, 3)
        c4, s4 = ca.cos(joint_angles[3]), ca.sin(joint_angles[3])
        rot_joint4[0, 0] = 1.0
        rot_joint4[1, 1] = c4;  rot_joint4[1, 2] = s4   # Note: positive because axis is "-1 0 0"
        rot_joint4[2, 1] = -s4; rot_joint4[2, 2] = c4
        
        rot_link4 = ca.mtimes(rot_link3, rot_joint4)
        
        # 10. manipulation_link4 to ee: pos="0.01 0 0"
        # 11. ee to ee_tool: pos="0.14 0 0"
        # Total: 0.01 + 0.14 = 0.15
        pos_ee_local = ca.MX([0.15, 0.0, 0.0])
        pos_ee = pos_link4 + ca.mtimes(rot_link4, pos_ee_local)
        
        # Apply base rotation and translation
        ee_pos_world = ca.mtimes(base_rotation_matrix, pos_ee) + base_pos
        ee_rot_world = ca.mtimes(base_rotation_matrix, rot_link4)
        
        # Convert rotation matrix to quaternion
        ee_quat = quaternion_from_rotation_matrix_ca(ee_rot_world)
        
        return ca.vertcat(ee_pos_world, ee_quat)
    
    def forward_kinematics(self, u):
        '''
        Direct transformation chain matching MuJoCo XML structure exactly
        input: u: [base_pos, base_euler(rpy), joint_angles]
        '''
        base_pos = u[0:3]
        base_euler = u[3:6]  # roll, pitch, yaw
        joint_angles = u[6:10]  # [joint1, joint2, joint3, joint4]

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
        pos_ee_local = np.array([0.15, 0.0, 0.0])
        pos_ee = pos_link4 + np.dot(rot_link4, pos_ee_local)
        
        # Apply base rotation and translation
        ee_pos_world = np.dot(base_rotation_matrix, pos_ee) + base_pos
        ee_rot_world = np.dot(base_rotation_matrix, rot_link4)
        
        # Convert rotation matrix to quaternion
        ee_quat = quaternion_from_rotation_matrix(ee_rot_world)
        
        return np.concatenate((ee_pos_world, ee_quat))


    def optimize(self, current_drone_state, target_ee_pos, target_ee_quat, drone_state_status):
        '''
        input:
        current_drone_state: [pos_x, pos_y, pos_z, w, x, y, z, theta1, theta2, theta_3, theta_4]
        target_ee_quat: [w, x, y, z]

        output:
        km_target: [pos_x, pos_y, pos_z, w, x, y, z, theta1, theta2, theta3]
        '''


        current_base_pos = current_drone_state[:3]
        # w, x, y, z
        current_base_quat = current_drone_state[3:7]
        current_base_euler = quaternion_to_rpy(current_base_quat, xyzw = False, degrees = False)
        # print("current_base_quat: ", current_base_quat)
        # print("current_base_euler: ", current_base_euler)
        current_manipulator_joints = current_drone_state[7:11]      # 4-element
        
        init_guess = np.concatenate([current_base_pos,
                                    current_base_euler,
                                    current_manipulator_joints])
        
        # predict_ee_state = self.forward_kinematics(init_guess)
        # predict_ee_pos = predict_ee_state[:3]
        # predict_ee_quat = predict_ee_state[3:7]
        # print("predict_ee_pos from current: ", predict_ee_pos)
        # print("predict_ee_quat from current: ", predict_ee_quat)


        self.opti.set_initial(self.u, init_guess)

        
        self.opti.set_value(self.obj_ee_pos_tar, target_ee_pos)
        self.opti.set_value(self.obj_ee_quat_tar, target_ee_quat)
        self.opti.set_value(self.obj_cur_joints, current_manipulator_joints)
        # Set the current base position so the cost can measure changes
        self.opti.set_value(self.obj_cur_base_pos, current_base_pos)
        
        if drone_state_status == TAKEOFF:
            # the arm is not moving
            # the abs between the arm joint diff is less than eps
            self.opti.set_value(self.p6_lb, self.takeoff_arm_joints[0] - EPS)
            self.opti.set_value(self.p6_ub, self.takeoff_arm_joints[0] + EPS)
            self.opti.set_value(self.p7_lb, self.takeoff_arm_joints[1] - EPS)
            self.opti.set_value(self.p7_ub, self.takeoff_arm_joints[1] + EPS)
            self.opti.set_value(self.p8_lb, self.takeoff_arm_joints[2] - EPS)
            self.opti.set_value(self.p8_ub, self.takeoff_arm_joints[2] + EPS)
            self.opti.set_value(self.p9_lb, self.takeoff_arm_joints[3] - EPS)
            self.opti.set_value(self.p9_ub, self.takeoff_arm_joints[3] + EPS)

        else:
            # Use joint limits from XML file
            self.opti.set_value(self.p6_lb, -0.1)
            self.opti.set_value(self.p6_ub, 130/180*np.pi)
            self.opti.set_value(self.p7_lb, -0.1)
            self.opti.set_value(self.p7_ub, 130/180*np.pi)
            self.opti.set_value(self.p8_lb, -30/180*np.pi)
            self.opti.set_value(self.p8_ub, 50/180*np.pi)
            self.opti.set_value(self.p9_lb, -360.0)  # Unlimited bounds for roll joint ±360°
            self.opti.set_value(self.p9_ub, 360.0)   # Unlimited bounds for roll joint ±360°

        try:
            sol = self.opti.solve()
            
            # Get the optimized joint angles
            km_target = sol.value(self.u)
            
            # verify the solution
            km_target = np.array([km_target[0], km_target[1], km_target[2], km_target[3], km_target[4], km_target[5], km_target[6], km_target[7], km_target[8], km_target[9]])
            
            predict_ee_state = self.forward_kinematics(km_target)

            predict_ee_pos = predict_ee_state[:3]
            predict_ee_orientation = predict_ee_state[3:]
            pose_error = np.sum((predict_ee_pos - target_ee_pos)**2)
            orientation_error = 1 - np.dot(predict_ee_orientation, target_ee_quat)**2
            
            if pose_error > 0.01:
                print(f"Warning: pose error is too large: {pose_error}")
                print(f"Target EE pos: {target_ee_pos}")
                print(f"Predicted EE pos: {predict_ee_pos}")
                print(f"Target EE quat: {target_ee_quat}")
                print(f"Predicted EE quat: {predict_ee_orientation}")

            # Yuanhang: rpy to quat
            km_target_quat = quaternion_from_rotation_matrix(rpy_to_rotation_matrix(km_target[3:6]))
            km_target = np.array([km_target[0], km_target[1], km_target[2], 
                                  km_target_quat[0], km_target_quat[1], km_target_quat[2], km_target_quat[3], 
                                  km_target[6], km_target[7], km_target[8], km_target[9]])
            return km_target
            
        except Exception as e:
            print(f"IK Optimization failed: {e}")
            print(f"Target EE pos: {target_ee_pos}")
            print(f"Target EE quat: {target_ee_quat}")
            print(f"Current drone state: {current_drone_state}")
            print(f"Init guess: {init_guess}")
            
            # Return None to indicate failure
            return None
    
    