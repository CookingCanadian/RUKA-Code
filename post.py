import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

data_dir = 'runs40N'


moment_runs = []
time_ref = None

for filename in os.listdir(data_dir):
    if not filename.endswith('.csv'):
        continue

    data = np.genfromtxt(os.path.join(data_dir, filename), delimiter=',', skip_header=1)

    time = data[:, 0]
    my   = data[:, 5]
    mz   = data[:, 6]

    moment_sum = np.abs(my) + np.abs(mz)

    if time_ref is None:
        time_ref = time
    else:
        if len(time) != len(time_ref):
            raise ValueError("err: time vector mismatch")

    moment_runs.append(moment_sum)

moment_runs = np.vstack(moment_runs)
moment_avg = np.mean(moment_runs, axis=0)

dt = np.mean(np.diff(time_ref))

window_times = [2]  # seconds 

plt.figure(figsize=(10, 6))
plt.plot(time_ref, moment_avg, alpha=0.3, label='Raw averaged signal')

for w in window_times:
    w_samples = int(w / dt)
    smooth = np.convolve(moment_avg,
                         np.ones(w_samples) / w_samples,
                         mode='same')
    plt.plot(time_ref, smooth, label=f'{w*1000:.0f} ms window')

plt.xlabel('Time (s)')
plt.ylabel('|My| + |Mz| (Nm)')
plt.title('Moving-Average Window Selection (40 N Input)')
plt.legend()
plt.grid(True)
plt.show()
