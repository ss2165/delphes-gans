import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

g = np.loadtxt('g_speed.txt')
pd = np.loadtxt('pd_speed.txt')
n = g[1:, 0]
gt = g[1:, 1]
pdt = pd[1:, 1]

(ag, bg), Ag = np.polyfit(np.log10(n), np.log10(gt), 1, cov=True)
(ap, bp), Ap = np.polyfit(np.log10(n), np.log10(pdt), 1, cov=True)

x = np.logspace(1, 5, 9)
gfit = np.power(10, ag*np.log10(x) + bg)
pdfit = np.power(10, ap*np.log10(x) + bp)

plt.loglog(x, gfit, '--', label='Generator fit')
plt.loglog(x, pdfit, label='PD fit')

plt.loglog(n, gt, 'x', label='Generator', ms=7)
plt.loglog(n, pdt, '+', label='PD', ms=7)

plt.xlabel(r'Number of events')
plt.ylabel(r'Time /s')

plt.xlim([10, None])
plt.ylim([1, None])
plt.grid()
plt.legend(loc='best')

print('G_var')
print(ag, bg, np.sqrt(np.diag(Ag)))
print('PD_var')
print(ap, bp, np.sqrt(np.diag(Ap)))
plt.show()


