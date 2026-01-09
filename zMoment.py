from VIS_FT import visFTDriver

import time, os
import numpy as np
from threading import Thread, Lock
import sys
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

if __name__ == "__main__":
    print("init VisualFT")

    visFT_driver = visFTDriver()
    visFT_ready = True

    try:
        visFT_driver.start()
        print("zeroing sensor")
        visFT_driver.zero()
        print("connected")
    except:
        visFT_ready = False
        input("Error: can't reach sensor")
        sys.exit()
    # end of sensor setup

    data = []
    dataTime = []

    count = 0
    iterations = 1000
    initialTime = time.time()

    with alive_bar(iterations, dual_line=True) as bar:
        while True:
            visFT_val = visFT_driver.read()
            
            data.append(visFT_val)
            dataTime.append(time.time() - initialTime)

            bar.text(f'visFT_val: {np.round(visFT_val, 3)}')
            bar()
            count += 1

            if count == iterations:
                print("reached max iterations")
                break

    visFT_driver.shutdown()

    # plotting sensor values
    data = np.array(data)
    data = np.abs(data)
    dataTime = np.array(dataTime)

    os.makedirs('runs40N', exist_ok=True) # for saving runs
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    combined_data = np.column_stack((dataTime, data))
    header = 'time,fx,fy,fz,mx,my,mz'
    np.savetxt(f'runs/visft_data_{timestamp}.csv', combined_data, delimiter=',', header=header, comments='')
    sns.set_theme()
    plt.figure(figsize=(10, 6))
    plt.plot(dataTime, data[:, 5])
    plt.xlabel('Time (s)')
    plt.ylabel('Z-Moment (Nm)')
    plt.title('Z-Moment over Time with 40 N Input Force (VC)')
    plt.grid(True)
    plt.savefig(f'runs/z_moment_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Data saved to runs/visft_data_{timestamp}.csv")