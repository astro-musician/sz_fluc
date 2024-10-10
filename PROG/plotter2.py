import numpy as np
import matplotlib.pyplot as plt
from cluster_SZ import cluster_info, mock_fluc

K = np.geomspace(0.001,1,20)
PS =mock_fluc.mock_PS(K,0.1,5,1)
plt.plot(K,PS)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()