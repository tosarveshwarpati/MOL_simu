import numpy as np
import scipy.signal
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True
const = np.loadtxt("OH_v+e.txt")
pos = np.loadtxt("OH_pos.txt")
data1 = np.loadtxt("NO.txt")
mask1 = (data1[:,0]>160) & (data1[:,0]<290)
data = data1[mask1]
lam = data[:,0]
I = data[:, 6]
T_e = const[:,0]
oe = const[:,1]
oexe = const[:,2]
wl = []

#up = [0,1,2,3,4,5,6,7,8,9,10]
up = [0,1,2,3,4]
down = [0,1,2,3]


def wntolam(wn):
	return (1/wn)*(10e6)


def del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, v1, v2):
	delta_T_e = abs(T_e2 - T_e1)
	G1 = oe1*(v1 + (1/2)) - oexe1*(v1 + (1/2))**2
	G2 = oe2*(v2 + (1/2)) - oexe2*(v2 + (1/2))**2
	delta_G = abs(G2 - G1)
	return delta_T_e + delta_G
index = [1, 2, 3]
for i in range(np.size(T_e)):
	for ind in index:
		T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[ind-1], T_e[ind], oe[ind-1], oe[ind], oexe[ind-1], oexe[ind]
		for j in up:
			for k in down:
				en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, j, k)
				Wl = wntolam(en)
				wl.append(Wl)
		
for xc in wl:
    plt.axvline(x=xc, color = 'r')

plt.ylabel(r'$I \rightarrow$')
plt.xlabel(r'$\lambda \longrightarrow$')
plt.plot(lam, I)


with open('position.txt', 'w') as f1:
	f1.write(str(wl))

