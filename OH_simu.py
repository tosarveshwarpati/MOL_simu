import numpy as np
import scipy.signal
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt
import sys
const = np.loadtxt("OH_v+e.txt")
T_e = const[:,0]
oe = const[:,1]
oexe = const[:,2]
p_en = []
q_en = []
r_en = []
p_wl = []
q_wl = []
r_wl = []
def wntolam(wn):
	return (1/wn)*(10e6)
def del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, v1, v2):
	delta_T_e = abs(T_e2 - T_e1)
	G1 = oe1*(v1 + (1/2)) - oexe1*(v1 + (1/2))**2
	G2 = oe2*(v2 + (1/2)) - oexe2*(v2 + (1/2))**2
	delta_G = abs(G2 - G1)
	return delta_T_e + delta_G
for i in range(np.size(T_e)):
	T_e1, T_e2 ,oe1, oe2, oexe1, oexe2 = T_e[i-1], T_e[i], oe[i-1], oe[i], oexe[i-1], oexe[i]
	P_en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, 0, 1)
	P_wl = wntolam(P_en)
	Q_en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, 0, 0)
	Q_wl = wntolam(Q_en)
	R_en = del_E(T_e1, T_e2 ,oe1, oe2, oexe1, oexe2, 1, 0)
	R_wl = wntolam(R_en)
	p_en.append(P_en)
	q_en.append(Q_en)
	r_en.append(R_en)
	p_wl.append(P_wl)
	q_wl.append(Q_wl)
	r_wl.append(R_wl)


