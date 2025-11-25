import numpy as np
import os
import sys
import scipy
import matplotlib.pyplot as plt
# import Rotation matrix
from scipy.spatial.transform import Rotation as R


from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import Bounds


def rotation_matrix_from_quaternion(quaternion):
    r = R.from_quat(quaternion)
    return r.as_dcm()

def quaternion_from_rotation_matrix(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()

def rotation_matrix_from_euler(euler):
    r = R.from_euler('zyx', euler, degrees=False)
    return r.as_dcm()

class ARM_6DOF():
    def __init__(self):

        init_theta = np.zeros(6)
        theta1, theta2, theta3, theta4, theta5, theta6 = init_theta

        self.DH_params = np.array([
                                    (np.pi/2,   0,   0.1625, theta1),
                                    (0,         -0.8620,    0.0, theta2),
                                    (0,     -0.7287,   0,   theta3),
                                    (np.pi/2, 0,     0.2010, theta4),
                                    (-np.pi/2,  0,     0.1593,   theta5),
                                    (0, 0,     0.1543, theta6)
                                ])
        
        tranformation_matrix_total = np.eye(4)
        for joint in range(6):
            DH_params = self.DH_params[joint]
            alpha, a, d, theta = DH_params
            T = self.transformation_matrix(alpha, a, d, theta)
            tranformation_matrix_total = np.dot(tranformation_matrix_total, T)
        self.transformation_matrix_total = tranformation_matrix_total

    def transformation_matrix(self, alpha, a, d, theta):
        
        transformation_matrix = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                                            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],    
                                            [0, np.sin(alpha), np.cos(alpha), d], 
                                        [0, 0, 0, 1]])
        # print(transformation_matrix)
        
        return transformation_matrix


    # def forward_kinematics(self, theta1, theta2, theta3, theta4, theta5, theta6):
    #     T = np.eye(4)
    #     for i in range(6):
    #         T = np.dot(T, self.DH_matrix(*self.DH_params[i]))
    #     return T

    def visualize(self, input_theta):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        base_joint_pos = np.array([[0, 0, 0, 1]]).T
        joint_pos = base_joint_pos
        tranformation_matrix_total = np.eye(4)
        
        # plot the joint as point
        ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red')
        # plot the frame
        new_frame = tranformation_matrix_total[:3, :3]
        axis_scale = 0.1
        ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 0]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 0]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 0]*axis_scale], color='red')
        ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 1]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 1]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 1]*axis_scale], color='green')
        ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 2]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 2]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 2]*axis_scale], color='blue')
            
        for i in range(6):
            
            
            # plot the link as line
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = input_theta[i]
            T = self.transformation_matrix(alpha, a, d, theta)
            tranformation_matrix_total = tranformation_matrix_total @ T
     
            
            next_joint_pos = np.dot(tranformation_matrix_total, base_joint_pos)
            # plot the line
            ax.plot([joint_pos[0], next_joint_pos[0]], [joint_pos[1], next_joint_pos[1]], [joint_pos[2], next_joint_pos[2]], color='blue')
            # plot the joint as point
            ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red')

            joint_pos = next_joint_pos
            # plot the joint as point
            ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red')
        
            # plot the frame
            new_frame = tranformation_matrix_total[:3, :3]
            axis_scale = 0.1
            ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 0]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 0]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 0]*axis_scale], color='red')
            ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 1]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 1]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 1]*axis_scale], color='green')
            ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 2]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 2]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 2]*axis_scale], color='blue')
            


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-3, 3])
        plt.show()

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        for i in range(6):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = joint_angles[i]
            T = np.dot(T, self.transformation_matrix(alpha, a, d, theta))
        base_pos = np.array([[0, 0, 0, 1]]).T
        ee_pos = np.dot(T, base_pos)
        ee_pos = ee_pos.flatten()
        orientation = quaternion_from_rotation_matrix(T[:3, :3])  # Convert rotation matrix to quaternion
        ee_state = np.concatenate((ee_pos[:3], orientation))
        
        return ee_state

    def inverse_kinematics(self, initial_guess, target_ee_state):
        def objective(theta):
            ee_state = self.forward_kinematics(theta)
            ee_pos = ee_state[:3]
            ee_orientation = ee_state[3:]
            target_ee_pos = target_ee_state[:3]
            target_ee_orientation = target_ee_state[3:]
            pose_error = np.sum((ee_pos - target_ee_pos[:3])**2)
            orientation_error = 1 - np.dot(ee_orientation, target_ee_orientation)**2
            error = pose_error + orientation_error
            return error
        
        # Initial guess for joint angles (in radians)
        joint_bounds = Bounds([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

        # Run the optimization
        result = minimize(objective, initial_guess,method='SLSQP', bounds=joint_bounds)

        # Check if the optimization was successful and print the result
        if result.success:
            print("Found joint angles: ", result.x)
        else:
            print("Optimization failed: ", result.message)
        return result.x


class ARM_7DOF():
    def __init__(self):

        init_theta = np.zeros(7)
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = init_theta

        # DH_params = (alpha, a, d, theta)
        self.DH_params = np.array([
                                    (np.pi/2,   0,      0.185, theta1),
                                    (-np.pi/2,  0,      0.0, theta2),
                                    (np.pi/2,   0,      0.280,   theta3),
                                    (-np.pi/2,  0,      0.0, theta4),
                                    (np.pi/2,  0,     0.27,   theta5),
                                    (-np.pi/2, 0.0,    0.0,     theta6),
                                    (np.pi/2,   0,     0.160, theta7)
                                ])
        
        tranformation_matrix_total = np.eye(4)
        for joint in range(7):
            DH_params = self.DH_params[joint]
            alpha, a, d, theta = DH_params
            T = self.transformation_matrix(alpha, a, d, theta)
            tranformation_matrix_total = np.dot(tranformation_matrix_total, T)
        self.transformation_matrix_total = tranformation_matrix_total

    def transformation_matrix(self, alpha, a, d, theta):
        
        transformation_matrix = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                                            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],    
                                            [0, np.sin(alpha), np.cos(alpha), d], 
                                        [0, 0, 0, 1]])
        # print(transformation_matrix)
        
        return transformation_matrix


    # def forward_kinematics(self, theta1, theta2, theta3, theta4, theta5, theta6):
    #     T = np.eye(4)
    #     for i in range(6):
    #         T = np.dot(T, self.DH_matrix(*self.DH_params[i]))
    #     return T

    def visualize(self, input_theta):
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        base_joint_pos = np.array([[0, 0, 0, 1]]).T
        joint_pos = base_joint_pos
        tranformation_matrix_total = np.eye(4)
        
        # plot the joint as point
        ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red')
        # plot the frame
        new_frame = tranformation_matrix_total[:3, :3]
        axis_scale = 0.1
        ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 0]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 0]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 0]*axis_scale], color='red')
        ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 1]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 1]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 1]*axis_scale], color='green')
        ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 2]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 2]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 2]*axis_scale], color='blue')
            
        for i in range(7):
            
            
            # plot the link as line
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = input_theta[i]
            T = self.transformation_matrix(alpha, a, d, theta)
            tranformation_matrix_total = tranformation_matrix_total @ T
     
            
            next_joint_pos = np.dot(tranformation_matrix_total, base_joint_pos)
            # plot the line
            ax.plot([joint_pos[0], next_joint_pos[0]], [joint_pos[1], next_joint_pos[1]], [joint_pos[2], next_joint_pos[2]], color='blue')
            # plot the joint as point
            ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red')

            joint_pos = next_joint_pos
            # plot the joint as point
            ax.scatter(joint_pos[0], joint_pos[1], joint_pos[2], color='red')
        
            # plot the frame
            new_frame = tranformation_matrix_total[:3, :3]
            axis_scale = 0.1
            ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 0]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 0]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 0]*axis_scale], color='red')
            ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 1]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 1]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 1]*axis_scale], color='green')
            ax.plot([joint_pos[0], joint_pos[0] + new_frame[0, 2]*axis_scale],[joint_pos[1], joint_pos[1] + new_frame[1, 2]*axis_scale],[joint_pos[2], joint_pos[2] + new_frame[2, 2]*axis_scale], color='blue')
            


        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-0.1, 1])
        plt.show()

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        for i in range(7):
            DH_params = self.DH_params[i]
            alpha, a, d, theta = DH_params
            theta = joint_angles[i]
            T = np.dot(T, self.transformation_matrix(alpha, a, d, theta))
        base_pos = np.array([[0, 0, 0, 1]]).T
        ee_pos = np.dot(T, base_pos)
        ee_pos = ee_pos.flatten()
        orientation = quaternion_from_rotation_matrix(T[:3, :3])  # Convert rotation matrix to quaternion
        ee_state = np.concatenate((ee_pos[:3], orientation))
        
        return ee_state

    def inverse_kinematics(self, initial_guess, target_ee_state):
        def objective(theta):
            ee_state = self.forward_kinematics(theta)
            ee_pos = ee_state[:3]
            ee_orientation = ee_state[3:]
            target_ee_pos = target_ee_state[:3]
            target_ee_orientation = target_ee_state[3:]
            pose_error = np.sum((ee_pos - target_ee_pos[:3])**2)
            orientation_error = 1 - np.dot(ee_orientation, target_ee_orientation)**2
            error = pose_error + orientation_error
            joint_change = np.sum((theta - initial_guess)**2)
            # error += joint_change*0.0001
            return error
        
        # Initial guess for joint angles (in radians)
        joint_bounds = Bounds([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

        # Run the optimization
        result = minimize(objective, initial_guess, method = "L-BFGS-B", bounds=joint_bounds)

        # Check if the optimization was successful and print the result
        if result.success:
            # print("Found joint angles: ", result.x)
            pass
        else:
            print("Optimization failed: ", result.message)
        return result.x





def main():

    arm = ARM_7DOF()
    theta_init = np.array([0., 0., 0., 0., 0., 0., 0.])
    T = arm.forward_kinematics(theta_init)
    print(T)
    
    # for degree in range(0,180,30):
    #     for joint_idx in range(7):
    #         theta = theta_init.copy()
    #         theta[joint_idx] += np.deg2rad(degree)
    #         print("Theta: ", theta)
    #         arm.visualize(theta)
    # arm.visualize(theta)


    # set 10 random theta
    max_pos_error = 0
    max_pos_error_joint_angles = []
    max_orientation_error = 0
    max_orientation_error_joint_angles = []
    for i in range(100):
        theta = np.random.rand(7)
        # scale the theta to -pi to pi
        # theta = theta * 2 * np.pi - np.pi
        print("Theta: ", theta)
        # arm.visualize(theta)
        gt_ee_state = arm.forward_kinematics(theta)
        print("EE pos: ", gt_ee_state)
        target_ee_state = gt_ee_state
        min_error = 1000
        pos_error_threshold = 1e-5

        for j in range(10):
            init_guess = np.random.rand(7) * 1.0*2*np.pi - np.pi
            ik_theta = arm.inverse_kinematics(init_guess, target_ee_state)
            ik_ee_state = arm.forward_kinematics(ik_theta)
            error = np.linalg.norm(target_ee_state[:3] - ik_ee_state[:3]) + 1 - np.dot(target_ee_state[3:], ik_ee_state[3:])**2
            pos_error = np.linalg.norm(target_ee_state[:3] - ik_ee_state[:3])
            if pos_error < pos_error_threshold:
                min_error_theta = ik_theta
                break
            if error < min_error:
                min_error = error
                min_error_theta = ik_theta
        ik_theta = min_error_theta
        print("IK theta: ", ik_theta)
        ik_ee_state = arm.forward_kinematics(ik_theta)
        print("IK calculated EE pos: ", ik_ee_state)
        print("pos error: ", np.linalg.norm(gt_ee_state[:3] - ik_ee_state[:3]))
        print("orientation error: ", 1 - np.dot(gt_ee_state[3:], ik_ee_state[3:])**2)
        if max_pos_error < np.linalg.norm(gt_ee_state[:3] - ik_ee_state[:3]):
            max_pos_error = np.linalg.norm(gt_ee_state[:3] - ik_ee_state[:3])
            max_pos_error_joint_angles = theta
        if max_orientation_error < 1 - np.dot(gt_ee_state[3:], ik_ee_state[3:])**2:
            max_orientation_error = 1 - np.dot(gt_ee_state[3:], ik_ee_state[3:])**2
            max_orientation_error_joint_angles = theta
    print("Max pos error: ", max_pos_error)
    print("Max orientation error: ", max_orientation_error)
    print("Max pos error joint angles: ", max_pos_error_joint_angles)
    print("Max orientation error joint angles: ", max_orientation_error_joint_angles)


if __name__ == "__main__":
    main()