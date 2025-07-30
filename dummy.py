import numpy as np
import matplotlib.pyplot as plt
from Code.Model import Model

d1 = d2 = 50  # [-20, 20]
V3 = 323.725000000*2
chop = 135
mu = [0, 1, 0]
jmin = 0
jmax = 10
T = 300
lims = [-500, 500]
shift = 0
width = 0.0016


F = 5.792292534889519
Fprime = -0.289405380715179
Ax = 0.678719102454549
Bz = 0.256805714117702
Cy = 0.186449993466079
consts = np.array([Cy, Bz, Ax])
uconsts = consts
Qx = 0.518612135474886
Qz = 0.165663518946693
V3 = 323.725000000 * 2
V6 = -3.277 * 2
V3p = 8.935
V3m = -7.361

nistconsts = np.array([0.19073, 0.25421, 0.59406])

chop = 135+54
DMS = Model(d1, d2, F, Fprime, 0*Qx, 0*Qz, nistconsts, nistconsts, V3, V6, V3p, V3m, chop, mu, jmin, jmax, T, lims, width, shift,
                stats=[1, 1, 1, 1])
DMS.newcalcspectrum(save = True, name = '6_quanta_J_0_70_no_coriolis')
spnocor,xnocor = DMS.plot(show = False)

DMS = Model(d1, d2, F, Fprime, Qx, Qz, consts, uconsts, V3, V6, V3p, V3m, chop, mu, jmin, jmax, T, lims, width, shift,
                stats=[1, 1, 1, 1])
# DMS = Model(d1, d2, F, Fprime, 0*Qx, 0*Qz, nistconsts, nistconsts, V3, V6, V3p, V3m, chop, mu, jmin, jmax, T, lims, width, shift,
#                 stats=[1, 1, 1, 1])
DMS.newcalcspectrum(save = True, name = '6_quanta_J_0_70_full')
spfull,xfull = DMS.plot(show = False)


plt.figure(figsize = (8,6))
# plt.plot(xnocor,spnocor/np.max(spnocor), label = 'Torsions, Coupling off', color = 'k')
# plt.plot(xfull,spfull/np.max(spfull), label = 'Torsion and Rotation', color = 'tab:red')
# plt.plot(rx,-rsp/np.max(rsp), label = 'Rotation', color = 'k')
plt.plot(xfull,spfull, label = 'Torsion and Rotation', color = 'tab:red')
plt.plot(xnocor,-spnocor, label = 'Rotation', color = 'k')
plt.legend(loc='upper right', fontsize = 18)
plt.xlim([-60,60])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Frequency [cm$^{-1}$]',fontsize=20)
plt.ylabel('Intensity',fontsize=20)
# plt.title('R Branch of $J\leq2$ Spectrum',fontsize=24)
plt.tight_layout()
plt.show()
plt.savefig('fullspectrum.png')
plt.close()


