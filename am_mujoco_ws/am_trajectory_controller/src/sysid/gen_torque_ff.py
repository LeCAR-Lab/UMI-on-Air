# gen_torque_ff.py
import numpy as np
import yaml
import pickle
import matplotlib.pyplot as plt
from scipy import signal
import os

def main():
    with open("config_gen.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    mag = cfg["sampling_magnitude"]
    dur = cfg["sampling_time"]
    total_time = cfg["total_time"]
    cutoff = cfg["low_pass_cutoff"]
    out_file = cfg["output_file"]
    fs = 100
    t_sample = np.arange(0, total_time, dur)
    noise = np.random.normal(0, mag, (len(t_sample), 3))
    noise[:, 0] = 0
    noise[:, 2] = 0
    # only inject noise for pitch
    
    t_total = np.arange(0, total_time, 1 / fs)
    
    torque = np.zeros((len(t_total), 3))
    for i in range(3):
        torque[:, i] = np.interp(t_total, t_sample, noise[:, i])
    
    b, a = signal.butter(4, cutoff / (fs / 2), btype='low')
    ff_data = signal.filtfilt(b, a, torque, axis=0)
    
    t = t_total
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(ff_data, f)
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
    for i in range(3):
        axs[i].plot(t, ff_data[:, i])
        axs[i].set_ylabel(f"Torque {i}")
        axs[i].grid()
    axs[-1].set_xlabel("Time [s]")
    axs[-1].set_xticks(np.arange(0, total_time + 1, 2))
    plt.tight_layout()
    plt.savefig(out_file.replace(".pkl", ".png"))

if __name__ == "__main__":
    main()