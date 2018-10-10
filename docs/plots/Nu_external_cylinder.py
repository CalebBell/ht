import matplotlib.pyplot as plt
import numpy as np
from ht.conv_external import conv_external_cylinder_methods, Nu_external_cylinder
styles = ['--', '-.', '-', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4']

Res = np.logspace(np.log10(10), np.log10(1E6), 1000)

Prs = np.array([[.7, 2, 6],
            [.1, .2, .4],
            [25, 100, 1000]])


f, axarr = plt.subplots(3, 3)

for Pr, axes in zip(Prs.ravel(), axarr.ravel()):
    for method, style in zip(conv_external_cylinder_methods, styles):
        Nus = [Nu_external_cylinder(Re=Re, Pr=Pr, Method=method) for Re in Res]
        axes.semilogx(Res, Nus, label=method) # + ', angle = ' + str(angle)
        
        axes.set_title(r'Pr = %g' %Pr)
        for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] +
             axes.get_xticklabels() + axes.get_yticklabels()):
            item.set_fontsize(6.5)
            
        ttl = axes.title.set_position([.5, .98])
    
plt.subplots_adjust(wspace=.35, hspace=.35)

f.suptitle('Comparison of available methods for external convection\n Reynolds Number (x) vs. Nusselt Number (y)')
plt.legend(loc='upper center', bbox_to_anchor=(1.5, 2.4))
plt.subplots_adjust(right=0.82, top=.85, bottom=.05)
#plt.show()



