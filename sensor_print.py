from VIS_FT import visFTDriver

import math, time, os
import numpy as np
from threading import Thread, Lock
import copy
import sys
from alive_progress import alive_bar
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Initiating VIS_FT sensor connection")
    
    # # Sensor Setup # #

    visft_driver=visFTDriver()
    visft_ready=True
    try:
        visft_driver.start()        
        print("Zeroing VIS_FT sensor...")
        visft_driver.zero()
        print("Connected to VIS_FT sensor!")
    except:
        input("Error: Can't reach VIS_FT sensor!")
        visft_ready=False
        sys.exit()

    # # END of Sensor Setup # #
    

    datVisFT_ = []
    datTime_ = []

    count = 0
    iterations = 1000
    initial_time = time.time()

    with alive_bar(iterations, dual_line=True) as bar:
        while (True):
            vis_ft_val = visft_driver.read()
            
            bar.text(f'vis_ft_val: {np.round(vis_ft_val,3)}')

            datVisFT_.append(vis_ft_val)
            datTime_.append(time.time() - initial_time)

            bar()
            count += 1
            if count == iterations:
                print("reached max iterations.")
                break

    visft_driver.shutdown()

    # Plotting sensor values

    datVisFT_ = np.array(datVisFT_)
    datTime_ = np.array(datTime_)

    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    axes = axes.flatten()
    titles = ["Force-X", "Force-Y", "Force-Z", "Moment-X", "Moment-Y", "Moment-Z"]
    ylabels = ["Force (N)", "Force (N)", "Force (N)", "Moment (Nm)", "Moment (Nm)", "Moment (Nm)"]

    for i in range(6):
        ax = axes[i]
        ax.plot(datTime_, datVisFT_[:, i])
        ax.set_title(titles[i])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabels[i])
        ax.grid(True)

    plt.tight_layout()
    plt.show()

