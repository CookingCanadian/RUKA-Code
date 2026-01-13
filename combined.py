import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file1 = '40N_averaged.csv'
file2 = '10N_averaged.csv'

# Load data
data1 = np.genfromtxt(file1, delimiter=',', skip_header=1)
time1 = data1[:, 0]
signal1 = data1[:, 1]

data2 = np.genfromtxt(file2, delimiter=',', skip_header=1)
time2 = data2[:, 0]
signal2 = data2[:, 1]

# Apply mask to limit both datasets to 7 s
mask1 = time1 <= 7
time1 = time1[mask1]
signal1 = signal1[mask1]

mask2 = time2 <= 7
time2 = time2[mask2]
signal2 = signal2[mask2]

sns.set_theme(rc={'axes.labelsize':20, 'legend.fontsize':14, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time1, signal1, color='#013ECD', linewidth=2, label='Filtered Torque 40 N')
plt.plot(time2, signal2, color='#E60039', linewidth=2, label='Filtered Torque 10 N')

plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)
plt.xlim(0, 7)
plt.ylim(0, max(np.max(signal1), np.max(signal2)) * 1.1)

plt.title('Filtered Torque Signal Comparison (10 N vs 40 N Input)', fontsize=20)
plt.tight_layout()
plt.show()
