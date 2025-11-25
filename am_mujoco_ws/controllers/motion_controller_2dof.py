import numpy as np
from . import pid_controller

class UAM2DoFMotionController():
    """
    Motion controller for the UAV and manipulator
    """
    def __init__(self, start_time = 0.0):
        self.controller = None
        print("start_time", start_time)
        # zero gravity
        self.uav_controller_px = pid_controller.PID(Kp=10.0, Ki=0.01, Kd=10.0, setpoint=0, start_time=start_time, output_limits=(None, None))
        self.uav_controller_py = pid_controller.PID(Kp=10.0, Ki=0.01, Kd=10.0, setpoint=0, start_time=start_time, output_limits=(None, None))
        self.uav_controller_pz = pid_controller.PID(Kp=10.0, Ki=1.0, Kd=10.0, setpoint=0, start_time=start_time, output_limits=(None, None))
        self.uav_controller_roll = pid_controller.PID(Kp=3.0, Ki=0.01, Kd=3.0, setpoint=0, start_time=start_time, output_limits=(-10, 10))
        self.uav_controller_pitch = pid_controller.PID(Kp=3.0, Ki=0.01, Kd=3.0, setpoint=0, start_time=start_time, output_limits=(-10, 10))
        self.uav_controller_yaw = pid_controller.PID(Kp=3.0, Ki=0.01, Kd=3.0, setpoint=0, start_time=start_time, output_limits=(-10, 10))
        self.manipulator_controller_link1_pitch = pid_controller.PID(Kp=1, Ki=0.01, Kd=0.15, setpoint=0, start_time=start_time, output_limits=(-1, 1))
        self.manipulator_controller_link2_pitch = pid_controller.PID(Kp=1, Ki=0.01, Kd=0.15, setpoint=0, start_time=start_time, output_limits=(-1, 1))
        # normal gravity
        # self.uav_controller_px = pid_controller.PID(Kp=400.0, Ki=200.0, Kd=100.0, setpoint=0, start_time=start_time, output_limits=(None, None))
        # self.uav_controller_py = pid_controller.PID(Kp=400.0, Ki=200.0, Kd=100.0, setpoint=0, start_time=start_time, output_limits=(None, None))
        # self.uav_controller_pz = pid_controller.PID(Kp=400.0, Ki=200.0, Kd=100.0, setpoint=0, start_time=start_time, output_limits=(None, None))
        # self.uav_controller_roll = pid_controller.PID(Kp=100.0, Ki=10.0, Kd=3.0, setpoint=0, start_time=start_time, output_limits=(-10, 10))
        # self.uav_controller_pitch = pid_controller.PID(Kp=100.0, Ki=10.0, Kd=3.0, setpoint=0, start_time=start_time, output_limits=(-10, 10))
        # self.uav_controller_yaw = pid_controller.PID(Kp=100.0, Ki=10.0, Kd=3.0, setpoint=0, start_time=start_time, output_limits=(-10, 10))
        # self.manipulator_controller_link1_pitch = pid_controller.PID(Kp=0.9, Ki = 0.5, Kd=0.2, setpoint=0, start_time=start_time, output_limits=(-1, 1))
        # self.manipulator_controller_link2_pitch = pid_controller.PID(Kp=0.5, Ki=0.5, Kd=0.1, setpoint=0, start_time=start_time, output_limits=(-1, 1))
        self.gripper_status_target = None

    def set_target(self, target):
        # target = [x, y, z, roll, pitch, yaw, manipualtor_base_yaw, manipulator_link1_pitch, manipulator_link2_pitch]
        self.uav_controller_px.setpoint(target[0])
        self.uav_controller_py.setpoint(target[1])
        self.uav_controller_pz.setpoint(target[2])
        self.uav_controller_roll.setpoint(target[3])
        self.uav_controller_pitch.setpoint(target[4])
        self.uav_controller_yaw.setpoint(target[5])
        self.manipulator_controller_link1_pitch.setpoint(target[6])
        self.manipulator_controller_link2_pitch.setpoint(target[7])
        if target.shape[0] >8:
            self.gripper_status_target = target[8]

    def get_control(self, current_state, current_time):
        # input: current state = [base_x, base_y, base_z,
        #                         base_roll, base_pitch, base_yaw,
        #                         link1_pitch, link2_pitch, 
        #                         gripper_left, gripper_right]
        #                         
        # print("time", current_time)
        # print("current_state", current_state)
        uav_control = np.array([self.uav_controller_px.get_control(current_state[0], current_time),
                                self.uav_controller_py.get_control(current_state[1], current_time),
                                self.uav_controller_pz.get_control(current_state[2], current_time),
                                self.uav_controller_roll.get_control(current_state[3], current_time),
                                self.uav_controller_pitch.get_control(current_state[4], current_time),
                                self.uav_controller_yaw.get_control(current_state[5], current_time)])

        manipulator_control = np.array([
                                        self.manipulator_controller_link1_pitch.get_control(current_state[6], current_time),
                                        self.manipulator_controller_link2_pitch.get_control(current_state[7], current_time)])

        if self.gripper_status_target is not None:
            if self.gripper_status_target <= 0.0:
                gripper_control = np.array([-0.01, -0.01])
            else:
                gripper_pos = -0.05 * self.gripper_status_target
                gripper_control = np.array([gripper_pos, gripper_pos])
        else:
            gripper_control = np.array([-0.01, -0.01])
        # print("gripper_control", gripper_control)
        control = np.concatenate((uav_control, manipulator_control, gripper_control))
        # print("control", control)
        return control