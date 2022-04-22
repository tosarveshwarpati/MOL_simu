import numpy as np
import scipy.signal
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True
const = np.loadtxt("NO_j+v+e_gamma.txt")
pos = np.loadtxt("OH_pos.txt")
data1 = np.loadtxt("NO.txt")
#posN2 = np.loadtxt("N2+B-X_simulation_SA_01.txt")
#posNO = np.loadtxt("NO_C-X_simulation_Voigt- 75%_lorentzian_SA01.txt")
#N2pos = posN2[:,0]/10
#N2int = posN2[:,1]
#NOpos = posNO[:,0]/10
#NOint = posNO[:,1]
mask1 = (data1[:,0]>110) & (data1[:,0]<290)
data = data1[mask1]
lam = data[:,0]
I = data[:, 6]
T_e = const[:,0]
oe = const[:,1]
oexe = const[:,2]
B = const[:,3]
D = const[:,4]
alpha = const[:,5]
wlAXR = []
wlAXP = []
wlAXQ = []
wlBX = []
wlCX = []

simu_datP = []
simu_datQ = []
simu_datR = []
wl = []
up = [0,1,2,3,4,5,6,7,8,9,10]
down = [0,1,2,3]
J1 = np.arange(0, 30, 1, dtype=int)
x_lam = np.arange(data[:,0][0],data[:,0][-1],0.1)

def wntolam(wn):
	return (1/wn)*(10e6)


def del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, v1, v2, B1, B2, D1, D2, j1, j2, alpha1, alpha2):
	delta_T_e = T_e1 - T_e2
	G1 = oe1*(v1 + (1/2)) - oexe1*(v1 + (1/2))**2
	G2 = oe2*(v2 + (1/2)) - oexe2*(v2 + (1/2))**2
	delta_G = G1 - G2
	B1_nu = B1 - alpha1*(v1 + (1/2))
	B2_nu = B2 - alpha2*(v2 + (1/2))
	F1 = B1_nu*(j1*(j1+1)) + D1*(j1**2)*(j1+1)**2
	F2 = B2_nu*(j2*(j2+1)) + D2*(j2**2)*(j2+1)**2
	delta_F = F1 - F2
	return delta_T_e + delta_G + delta_F
# A>X, B>A, C>B
for rot_qu in J1:
	for i in range(np.size(T_e)-1):
		T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, B1, B2, D1, D2, alpha1, alpha2 = T_e[i], T_e[i+1], oe[i], oe[i+1], oexe[i], oexe[i+1], B[i], B[i+1], D[i], D[i+1], alpha[i], alpha[i+1]
		for j in up:
			for k in down:
				j1, j2 = rot_qu, rot_qu - 1
				en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j, B1, B2, D1, D2, j1, j2, alpha1, alpha2)
				Wl = wntolam(en)
				dat = Wl,  T_e1,  T_e2,  oe1,  oe2,  oexe1,  oexe2,  k,  j, B1, B2, D1, D2, j1, j2, "R"
				wlAXR.append(Wl)
				simu_datR.append(dat)
for rot_qu in J1:
	for i in range(np.size(T_e)-1):
		T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, B1, B2, D1, D2, alpha1, alpha2 = T_e[i], T_e[i+1], oe[i], oe[i+1], oexe[i], oexe[i+1], B[i], B[i+1], D[i], D[i+1], alpha[i], alpha[i+1]
		for j in up:
			for k in down:
				j1, j2 = rot_qu, rot_qu
				en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j, B1, B2, D1, D2, j1, j2, alpha1, alpha2)
				Wl = wntolam(en)
				dat = Wl,  T_e1,  T_e2,  oe1,  oe2,  oexe1,  oexe2,  k,  j, B1, B2, D1, D2, j1, j2, "Q"
				wlAXQ.append(Wl)
				simu_datQ.append(dat)
for rot_qu in J1:
	for i in range(np.size(T_e)-1):
		T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, B1, B2, D1, D2, alpha1, alpha2 = T_e[i], T_e[i+1], oe[i], oe[i+1], oexe[i], oexe[i+1], B[i], B[i+1], D[i], D[i+1], alpha[i], alpha[i+1]
		for j in up:
			for k in down:
				j1, j2 = rot_qu, rot_qu + 1
				en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j, B1, B2, D1, D2, j1, j2, alpha1, alpha2)
				Wl = wntolam(en)
				dat = Wl,  T_e1,  T_e2,  oe1,  oe2,  oexe1,  oexe2,  k,  j, B1, B2, D1, D2, j1, j2, "P"
				wlAXP.append(Wl)
				simu_datP.append(dat)
				
#for i in range(np.size(T_e)-2):
#	T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[i], T_e[i+2], oe[i], oe[i+2], oexe[i], oexe[i+2]
#	for j in up:
#		for k in down:
#			en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j)
#			Wl = wntolam(en)
#			wlBX.append(Wl)

#for i in range(np.size(T_e)-3):
#	T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[i], T_e[i+3], oe[i], oe[i+3], oexe[i], oexe[i+3]
#	for j in up:
#		for k in down:
#			en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, k, j)
#			Wl = wntolam(en)
#			wlCX.append(Wl)
for xc in wlAXP:
    plt.axvline(x=xc, color = 'g')
for xc in wlAXQ:
    plt.axvline(x=xc, color = 'k', linestyle=':')
for xc in wlAXR:
    plt.axvline(x=xc, color = 'b', linestyle='--')#label = "$ A \rightarrow X, B \rightarrow A, C \rightarrow B$")
#for xc in wlBX:
#    plt.axvline(x=xc, color = 'g')
#for xc in wlCX:
#    plt.axvline(x=xc, color = 'b')
#for xc in NOpos:
#    plt.axvline(x=xc, color = 'g')
wl = np.stack((wlAXP, wlAXQ, wlAXR), axis = -1)
#wl.append(wlAXP)
#wl.append(wlAXQ)
#wl.append(wlAXR)
#wl.append(wlBX)
#wl.append(wlCX)

def inst_gaussian(l, Intensity, width, lam0):
	return ((Intensity)/(np.sqrt(2*np.pi*width)))*np.exp(-(l-lam0)**2/(2*width**2))


plt.ylabel(r'$I \rightarrow$')
plt.xlabel(r'$\lambda \longrightarrow$')
#plt.plot(lam, I, color = 'b', label = "no $N_2$ flow")
#plt.plot(lam, data[:, 6], color = 'y', label = "low $N_2$ flow")
#plt.plot(lam, data[:, 4], color = 'c', label = "moderate $N_2$ flow")
plt.plot(lam, data[:, 3], color = 'm', label = "Ut 2022 03 15")
plt.legend()

with open('position.txt', 'w') as f1:
	f1.write(str(wl))

with open('Output.txt', 'w') as f2:
	f2.write(str(simu_datP))


def Gauss(x_lam, amp_g, cen, width_g):
	return ((amp_g)*np.exp(-((x_lam-cen)**2)/(2*width_g**2)))


width = .8
column = 3

def Int(x_lam, width, lam, wl):
	Int = 0
	for i1 in wl[0]:
		I_index = np.searchsorted(lam, i1)
		I1 = data[:,column][I_index-1]
		Int1 = Gauss(x_lam, I1, i1, width)
		Int = Int + Int1
	plt.plot(x_lam, Int, color = 'k', label = "simulated")
	plt.legend()
	return plt.show()


