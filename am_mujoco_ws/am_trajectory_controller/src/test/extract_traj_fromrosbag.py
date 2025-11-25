import os
import sys
import time
import rospy
import rosbag
import numpy as np
import matplotlib.pyplot as plt
import csv



class TrajectoryExtractor:
    def __init__(self):
        pass

    def load_rosbag(self, bag_file):
        bag = rosbag.Bag(bag_file)
        print("loading done!")

        ee_target_pos = []
        t0 = None
        for topic, msg, t in bag.read_messages():        
            t = t.to_sec()
            if t0 is None:
                t0 = t
            t -= t0
            print(t)
            if topic == "/ee_tracking_target":
                temp_ee_target_pos = [t, msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
                ee_target_pos.append(temp_ee_target_pos)
        
        bag.close()
        print("reading messages done!")
        ee_target_pos = np.array(ee_target_pos)
        self.ee_target_pos = ee_target_pos
        print("ee_target_pos shape: ", self.ee_target_pos.shape)
        
        start_time = 88
        end_time = 183
        start_index = np.where(ee_target_pos[:, 0] >= start_time)[0][0]
        end_index = np.where(ee_target_pos[:, 0] <= end_time)[0][-1]
        self.ee_target_pos = ee_target_pos[start_index:end_index, :]
        # subtract the initial time
        self.ee_target_pos[:, 0] -= self.ee_target_pos[0, 0]

        return

    def visualize_raw_data(self):
        # create a 3*1 subplot
        fig, axs = plt.subplots(3, 1)
        fig.suptitle('ee_target_pos')
        # plot x, y, z
        axs[0].plot(self.ee_target_pos[:, 0], self.ee_target_pos[:, 1])
        axs[0].set_title('x')
        axs[1].plot(self.ee_target_pos[:, 0], self.ee_target_pos[:, 2])
        axs[1].set_title('y')
        axs[2].plot(self.ee_target_pos[:, 0], self.ee_target_pos[:, 3])
        axs[2].set_title('z')
        # show the plot
        plt.show()

    def save_ee_target_pos(self, save_data_path):
        dt = 0.05
        new_timestamp = np.arange(0, self.ee_target_pos[-1, 0], dt)
        # interpolate the data
        new_ee_target_pos = np.zeros((new_timestamp.shape[0], 4))
        for i in range(4):
            new_ee_target_pos[:, i] = np.interp(new_timestamp, self.ee_target_pos[:, 0], self.ee_target_pos[:, i])
        # save the data as a csv file
        with open(save_data_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])
            for i in range(new_timestamp.shape[0]):
                # save the data with 6 decimal places
                writer.writerow([
                                f"{new_timestamp[i]:.6f}",
                                f"{new_ee_target_pos[i, 1]:.6f}",
                                f"{new_ee_target_pos[i, 2]:.6f}",
                                f"{new_ee_target_pos[i, 3]:.6f}",
                                f"{0.0:.6f}", f"{0.0:.6f}", f"{0.0:.6f}", f"{1.0:.6f}"
                            ])
            print("save done!")


def main():
    rosbag_file = "/home/xiaofeng/workDisk/work/AerialManipulation/Project_Teleoperation/data/rosbag/20250404/rosbag/2025-04-04-03-16-22.bag"
    extractor = TrajectoryExtractor()
    extractor.load_rosbag(rosbag_file)
    extractor.visualize_raw_data()
    save_data_path = "./target_ee_pos.csv"
    extractor.save_ee_target_pos(save_data_path)


if __name__ == "__main__":
    main()