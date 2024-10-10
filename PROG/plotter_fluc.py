import numpy as np
import matplotlib.pyplot as plt

norm=1e-5
pix_inj=200
pix_dis=1
k_inj=1/pix_inj
k_dis=1/pix_dis

def kolmogorov(k):
    k = np.clip(k,a_min=1e-3,a_max=np.inf)
    return norm*np.exp(-k_inj/k)*np.exp(-k/k_dis)*k**(-11/3)

K = np.geomspace(1/500,1,500)
PS = kolmogorov(K)
ymin=np.min(PS)
ymax=np.max(PS)
inj_line_x = np.array([k_inj,k_inj])
inj_line_y = np.array([ymin,ymax])
dis_line_x = np.array([k_dis,k_dis])
dis_line_y = np.array([ymin,ymax])

font = {
    'size':12,
    'color':'black',
    'weight':'semibold',
}

fontcom = {
    'size':10,
    'color':'red',
    'weight':'semibold',
}

plt.figure(1)
plt.plot(K,PS,inj_line_x,inj_line_y,'r--',dis_line_x,dis_line_y,'r--')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('k',fontdict=font)
plt.ylabel('Power spectrum',fontdict=font)
plt.grid()
plt.text(1.1*k_inj,ymin,'Injection',fontdict=fontcom)
plt.text(1.1*k_dis,ymin,'Dissipation',fontdict=fontcom)
plt.text(k_inj**(2/3)*k_dis**(1/3),ymin**(3/4)*ymax**(1/4),r'$\mathbf{PS(k) \propto k^{-11/3}}$',fontdict=fontcom,size=12)
plt.title('Spectre de Kolmogorov',fontdict=font)

fluc_map = True
if fluc_map:

    length=200
    X, Y, Z = np.arange(length), np.arange(length), np.arange(length)
    X, Y, Z = np.meshgrid(X,Y,Z)
    kx, ky, kz = np.fft.fftfreq(length), np.fft.fftfreq(length), np.fft.fftfreq(length)
    kx, ky, kz = np.meshgrid(kx,ky,kz)
    knorm = np.sqrt(kx**2+ky**2+kz**2)
    PSgrid = kolmogorov(knorm)

    white_noise = np.random.normal(size=(length,length,length))
    noise = np.real(np.fft.ifftn(np.fft.fftn(white_noise,axes=(0,1,2))*np.sqrt(PSgrid),axes=(0,1,2)))
    noise_map = noise[0,:,:]

    plt.figure(2)
    plt.imshow(noise_map,cmap='Blues')
    plt.title('Champ de vitesse : inj='+str(pix_inj)+'px, dis='+str(pix_dis)+'px',fontdict=font)
    plt.colorbar()

plt.show()