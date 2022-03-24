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
posN2 = np.loadtxt("N2+B-X_simulation_SA_01.txt")
posNO = np.loadtxt("NO_C-X_simulation_Voigt- 75%_lorentzian_SA01.txt")
N2pos = posN2[:,0]/10
N2int = posN2[:,1]
NOpos = posNO[:,0]/10
NOint = posNO[:,1]
mask1 = (data1[:,0]>110) & (data1[:,0]<290)
data = data1[mask1]
lam = data[:,0]
I = data[:, 6]
T_e = const[:,0]
oe = const[:,1]
oexe = const[:,2]
wlAX = []
wlBX = []
wlCX = []
wl = []
up = [0,1,2,3,4,5,6,7,8,9,10]
down = [0,1,2,3]


def wntolam(wn):
	return (1/wn)*(10e6)


def del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, v1, v2):
	delta_T_e = T_e1 - T_e2
	G1 = oe1*(v1 + (1/2)) - oexe1*(v1 + (1/2))**2
	G2 = oe2*(v2 + (1/2)) - oexe2*(v2 + (1/2))**2
	delta_G = G1 - G2
	return delta_T_e + delta_G
# A>X, B>A, C>B
for i in range(np.size(T_e)-1):
	T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[i], T_e[i+1], oe[i], oe[i+1], oexe[i], oexe[i+1]
	for j in up:
		for k in down:
			en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j)
			Wl = wntolam(en)
			wlAX.append(Wl)

for i in range(np.size(T_e)-2):
	T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[i], T_e[i+2], oe[i], oe[i+2], oexe[i], oexe[i+2]
	for j in up:
		for k in down:
			en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j)
			Wl = wntolam(en)
			wlBX.append(Wl)

for i in range(np.size(T_e)-3):
	T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[i], T_e[i+3], oe[i], oe[i+3], oexe[i], oexe[i+3]
	for j in up:
		for k in down:
			en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j)
			Wl = wntolam(en)
			wlCX.append(Wl)
for xc in wlAX:
    plt.axvline(x=xc, color = 'r')#label = "$ A \rightarrow X, B \rightarrow A, C \rightarrow B$")
for xc in wlBX:
    plt.axvline(x=xc, color = 'g')
for xc in wlCX:
    plt.axvline(x=xc, color = 'b')
#for xc in NOpos:
#    plt.axvline(x=xc, color = 'g')
wl.append(wlAX)
wl.append(wlBX)
wl.append(wlCX)

plt.ylabel(r'$I \rightarrow$')
plt.xlabel(r'$\lambda \longrightarrow$')
plt.plot(lam, I, color = 'k', label = "no $N_2$ flow")
plt.plot(lam, data[:, 5], color = 'y', label = "low $N_2$ flow")
plt.plot(lam, data[:, 4], color = 'c', label = "moderate $N_2$ flow")
plt.plot(lam, data[:, 3], color = 'm', label = "xxxxxxxxxxxx")
plt.legend()

with open('position.txt', 'w') as f1:
	f1.write(str(wl))
