import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'runs10N'

window_time = 2.0  # smoothing window in seconds
t_min = 4.0        # start of plateau
t_max = 6.0       # end of plateau

moment_runs = []
time_ref = None

for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'):
        continue
    data = np.genfromtxt(os.path.join(data_dir, filename), delimiter=',', skip_header=1)
    time = data[:, 0]
    mx = data[:, 4]
    my = data[:, 5]
    mz = data[:, 6]
    moment_sum = np.abs(mx) + np.abs(mz)
    if time_ref is None:
        time_ref = time
    else:
        if len(time) != len(time_ref):
            raise ValueError(f"Time vector mismatch in {filename}")
    moment_runs.append(moment_sum)

moment_runs = np.vstack(moment_runs)
moment_avg = np.mean(moment_runs, axis=0)

dt = np.mean(np.diff(time_ref))
window_samples = int(window_time / dt)
moment_smooth = np.convolve(moment_avg, np.ones(window_samples) / window_samples, mode='same')

time_mask = (time_ref >= t_min) & (time_ref <= t_max)
plateau_moments = moment_smooth[time_mask]

plateau_mean = np.mean(plateau_moments)
plateau_max = np.max(plateau_moments)
plateau_std = np.std(plateau_moments)

print(f"Plateau mean: {plateau_mean:.3f} Nm")
print(f"Plateau max:  {plateau_max:.3f} Nm")
print(f"Plateau std:  {plateau_std:.3f} Nm")

averaged_data = np.column_stack((time_ref, moment_smooth))
np.savetxt('10N_averaged.csv', averaged_data, delimiter=',', header='time,Filtered_Torque_Signal', comments='')
sns.set_theme(rc={'axes.labelsize':20, 'legend.fontsize':14, 'xtick.labelsize': 16, 'ytick.labelsize': 16})
plt.figure(figsize=(10, 6))
plt.plot(time_ref, moment_smooth, color='#013ECD', linewidth=2, label='Filtered torque signal')
plt.axvspan(t_min, t_max, color='#7C878E', alpha=0.3, label='Plateau interval')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)
plt.xlim(0, 7)
plt.show()
