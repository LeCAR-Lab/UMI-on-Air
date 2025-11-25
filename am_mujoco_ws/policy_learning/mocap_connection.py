import socket
import threading
import numpy as np
from collections import deque

# Smoothing parameters
SMOOTHING_WINDOW = 15
SMOOTHING_ALPHA = 0.1  # Exponential smoothing factor (0 = no smoothing, 1 = no history)

# Global shared buffer for pose
last_pose = {
    "ee_pos": np.array([0.0, 0.0, 0.0]),    
    "ee_quat": np.array([0.0, 0.0, 0.0, 1.0]),  # XYZW format (identity quaternion)
    "gripper": 1.0  # Default gripper value (fully open)
}

# History buffers for smoothing
pose_history = {
    "ee_pos": deque(maxlen=SMOOTHING_WINDOW),
    "ee_quat": deque(maxlen=SMOOTHING_WINDOW),
    "gripper": deque(maxlen=SMOOTHING_WINDOW)
}

# Calibration system for relative positioning
calibration_data = {
    "calibrated": False,
    "reference_mocap_pos": None,
    "reference_ee_pos": np.array([0.0, 0.0, 1.2]),  # Default starting position (UMI/PnP/Rotate Valve)
    "raw_mocap_pos": np.array([0.0, 0.0, 0.0]),     # Current raw mocap position
}

# Connection state
connection_active = False
connection_event = threading.Event()

# -----------------------------------------------------
# Debug flag – set to True for verbose mocap diagnostics
# -----------------------------------------------------
DEBUG_MOCAP = False

def _dbg(*args, **kwargs):
    """Print only when DEBUG_MOCAP is True."""
    if DEBUG_MOCAP:
        print(*args, **kwargs)

def handle_connection(conn):
    """Handle incoming mocap data from TCP connection"""
    global last_pose, connection_active, calibration_data
    connection_active = True
    connection_event.set()

    _dbg("🧠 handle_connection thread started")
    buffer = b""
    while True:
        data = conn.recv(1024)
        _dbg(f"📦 Raw data received: {data}")
        if not data:
            print("⚠️ No data, closing connection")
            break

        buffer += data
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            decoded = line.decode().strip()
            _dbg(f"🧾 Line received: '{decoded}'")

            values = list(map(float, decoded.split(",")))
            _dbg(f"🔍 Parsed floats: {values}")
            if len(values) == 9:  # Now expecting 9 values: ID + 3 pos + 4 quat + 1 gripper
                id_val, *vals = values
                if int(id_val) == 1095:
                    # Store raw mocap position for calibration
                    raw_pos = np.array(vals[0:3])
                    raw_quat = np.array(vals[3:7])  # OptiTrack quaternion in XYZW format
                    raw_gripper = vals[7]  # Gripper value
                    calibration_data["raw_mocap_pos"] = raw_pos
                    
                    if not calibration_data["calibrated"]:
                        # First frame - use as calibration
                        calibration_data["reference_mocap_pos"] = raw_pos.copy()
                        calibration_data["calibrated"] = True
                        _dbg(f"🎯 Calibrated! Reference mocap pos: {raw_pos}")
                        # Set the ee_pos to the reference position for first frame
                        ee_pos = calibration_data["reference_ee_pos"].copy()
                    else:
                        # Subsequent frames - calculate relative position
                        mocap_delta = raw_pos - calibration_data["reference_mocap_pos"]
                        ee_pos = calibration_data["reference_ee_pos"] + mocap_delta
                        _dbg(f"📍 Relative pos - Mocap delta: {mocap_delta}, EE pos: {ee_pos}")
                    
                    # Convert from mocap XYZW to MuJoCo WXYZ format
                    mujoco_quat = np.array([raw_quat[3], raw_quat[0], raw_quat[1], raw_quat[2]])
                    
                    # Scale gripper from [0.05, 0.13] to [1.0, 0.0] (inverted), clamped between 0.0 and 1.0
                    # Inverted linear scaling: 1.0 - (value - min) / (max - min)
                    gripper_scaled = 1.0 - (raw_gripper - 0.05) / (0.13 - 0.05)
                    # Clamp between 0.0 and 1.0
                    gripper_scaled = max(0.0, min(1.0, gripper_scaled))
                    
                    # Apply smoothing to all values
                    smoothed_ee_pos = apply_smoothing(ee_pos, "ee_pos")
                    smoothed_ee_quat = apply_smoothing(mujoco_quat, "ee_quat")
                    smoothed_gripper = apply_smoothing(gripper_scaled, "gripper")
                    
                    # Store smoothed values
                    last_pose["ee_pos"] = smoothed_ee_pos
                    last_pose["ee_quat"] = smoothed_ee_quat
                    last_pose["gripper"] = smoothed_gripper
                    
                    _dbg(f"🔄 Raw mocap quat (XYZW): {raw_quat}")
                    _dbg(f"🔄 Final MuJoCo quat (WXYZ): {mujoco_quat}")
                    _dbg(f"🤏 Raw gripper: {raw_gripper}, Scaled gripper: {gripper_scaled}")
                    _dbg(f"📊 Smoothed - Pos: {smoothed_ee_pos}, Gripper: {smoothed_gripper:.3f}")
                    
                else:
                    _dbg(f"⚠️ Ignored ID {id_val}")
            else:
                _dbg(f"❌ Invalid value length ({len(values)}): expected 9, got {len(values)}")

    conn.close()
    connection_active = False
    print("❌ Client disconnected.")

def start_server():
    """Start TCP server to listen for mocap connections"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("localhost", 5500))
    server.listen(1)
    print("TCP server listening on localhost:5500")
    
    while True:
        print("Waiting for client connection...")
        conn, addr = server.accept()
        print(f"✅ Client connected from {addr}")
        threading.Thread(target=handle_connection, args=(conn,), daemon=True).start()

def initialize_mocap_connection():
    """Initialize the mocap connection server"""
    threading.Thread(target=start_server, daemon=True).start()
    return connection_event

def get_mocap_data():
    """Get current mocap data as action vector"""
    ee_pos = last_pose["ee_pos"]
    ee_quat = last_pose["ee_quat"]
    gripper_ctrl = last_pose.get("gripper", 1.0)  # Default to 1.0 if not set
    return np.concatenate([ee_pos, ee_quat, [gripper_ctrl]])

def reset_calibration():
    """Reset calibration to recalibrate on next mocap frame"""
    global calibration_data
    calibration_data["calibrated"] = False
    calibration_data["reference_mocap_pos"] = None
    print("🔄 Calibration reset - next mocap frame will be used as reference")

def get_calibration_status():
    """Get current calibration status and data"""
    return {
        "calibrated": calibration_data["calibrated"],
        "reference_mocap_pos": calibration_data["reference_mocap_pos"],
        "reference_ee_pos": calibration_data["reference_ee_pos"],
        "current_raw_mocap": calibration_data["raw_mocap_pos"],
        "current_ee_pos": last_pose["ee_pos"]
    }

def apply_smoothing(new_value, history_key):
    """Apply weighted moving average smoothing to a value"""
    history = pose_history[history_key]
    history.append(new_value)
    
    if len(history) == 1:
        # First value, no smoothing needed
        return new_value
    
    # Apply exponential weighted average
    # More recent values get higher weights
    weights = np.array([SMOOTHING_ALPHA * (1 - SMOOTHING_ALPHA) ** i for i in range(len(history))])
    weights = weights[::-1]  # Reverse so most recent gets highest weight
    weights = weights / np.sum(weights)  # Normalize weights
    
    if isinstance(new_value, np.ndarray):
        # For arrays (position, quaternion), apply weights to each element
        smoothed = np.zeros_like(new_value)
        for i, (val, weight) in enumerate(zip(history, weights)):
            smoothed += val * weight
        return smoothed
    else:
        # For scalars (gripper)
        return sum(val * weight for val, weight in zip(history, weights)) 