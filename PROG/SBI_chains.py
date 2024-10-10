import numpy as np
import matplotlib.pyplot as plt
import pickle

amas='A85'
n_R500_high = 2
n_R500_low = 0

with open('fluc/'+amas+'_chains_'+str(n_R500_low)+'-'+str(n_R500_high)+'R500.pkl','rb') as f:
    chains = pickle.load(f)

print(np.shape(chains))

X = np.arange(5000)

plt.figure(1)
for i in range(np.shape(chains)[1]):
    plt.plot(X,chains[:,i,0])
plt.grid()
plt.title(r'$k_{inj}$')

plt.figure(2)
for i in range(np.shape(chains)[1]):
    plt.plot(X,chains[:,i,1])
plt.grid()
plt.title(r'slope')

plt.figure(3)
for i in range(np.shape(chains)[1]):
    plt.plot(X,chains[:,i,2])
plt.grid()
plt.title(r'$\log (norm)$')

plt.show()