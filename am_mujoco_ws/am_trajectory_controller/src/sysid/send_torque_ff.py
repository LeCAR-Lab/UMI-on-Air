# publish_torque_ff.py
import rospy
import yaml
import pickle
from geometry_msgs.msg import Vector3
from mav_msgs.msg import RateThrust

def main():
    with open("config_send.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    file_name = cfg["torque_ff_file"]
    with open(file_name, "rb") as f:
        ff_data = pickle.load(f)


    print(f"Publishing torque feedforward from {file_name}")
    print(f"total_time: {len(ff_data) / 100:.2f} s")
    print(f"max torque: {ff_data.max(axis=0)}")
    print(f"min torque: {ff_data.min(axis=0)}")
    
    pub = rospy.Publisher("/wrench_controller/torque_ff", RateThrust, queue_size=10)
    rospy.init_node("torque_ff_publisher", disable_signals=True)
    idx = 0
    n = ff_data.shape[0]
    
    print("Press Enter to publish the next torque feedforward")
    input()
    
    rate = rospy.Rate(100)
    global hz_cnt
    hz_cnt = 0
    
    def hz_cb(event):
        global hz_cnt
        print(f"torque_ff hz: {hz_cnt}")
        hz_cnt = 0
        
    rospy.Timer(rospy.Duration(1), hz_cb)
    
    while not rospy.is_shutdown():
        try:
            msg = RateThrust()
            msg.header.stamp = rospy.Time.now()
            if idx < n:
                torque = ff_data[idx]
                msg.angular_rates = Vector3(*torque)
                idx += 1
            else:
                msg.angular_rates = Vector3(0.0, 0.0, 0.0)
            
            hz_cnt += 1
            pub.publish(msg)
            rate.sleep()
                
        except KeyboardInterrupt:
            msg = RateThrust()
            msg.header.stamp = rospy.Time.now()
            msg.angular_rates = Vector3(0.0, 0.0, 0.0)
            
            pub.publish(msg)
            print("Received keyboard interrupt, stopping, publishing zero torque and exiting")
            break

if __name__ == "__main__":
    main()