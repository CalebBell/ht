# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import division
from ht import *
from ht.conv_free_enclosed import Nu_Nusselt_Rayleigh_Holling_Herwig
import numpy as np

from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import bisplrep, UnivariateSpline


def test_Nu_Nusselt_Rayleigh_Holling_Herwig():
    Ras = [10.0**n for n in range(5, 16)]
    Nus_expect = [4.566, 8.123, 15.689, 31.526, 64.668, 134.135, 279.957, 586.404, 1230.938, 2587.421, 5443.761]
    Nus_calc = [round(Nu_Nusselt_Rayleigh_Holling_Herwig(1, Gr), 3) for Gr in Ras]
    assert_allclose(Nus_expect, Nus_calc)

    assert 1 == Nu_Nusselt_Rayleigh_Holling_Herwig(1, 100, buoyancy=True)
    assert 1 == Nu_Nusselt_Rayleigh_Holling_Herwig(1, 100, buoyancy=False)
    

def test_Nu_Nusselt_Rayleigh_Probert():
    Nu =  Nu_Nusselt_Rayleigh_Probert(5.54, 3.21e8, buoyancy=True)
    assert_allclose(Nu, 111.46181048289132)
    
    
    # Test the boundary
    Nu = Nu_Nusselt_Rayleigh_Probert(1, 2.19999999999999e4, buoyancy=True)
    assert_allclose(Nu, 2.5331972341122833)
    
    Nu = Nu_Nusselt_Rayleigh_Probert(1, 2.2e4, buoyancy=True)
    assert_allclose(Nu, 2.577876184202956)
    
    assert 1 == Nu_Nusselt_Rayleigh_Probert(1, 100, buoyancy=True)
    assert 1 == Nu_Nusselt_Rayleigh_Probert(1, 100, buoyancy=False)
    
    
def test_Rac_Nusselt_Rayleigh():
    for Rac_expect, insulation in zip([3011480.513694726, 9802960], [True, False]):
        for L in (8, 9, 100):
            W_L = .125
            Rac = Rac_Nusselt_Rayleigh(1, L, W_L*L, insulation)
            assert_allclose(Rac, Rac_expect)
            

def test_Rac_Nusselt_Rayleigh_disk():
    assert_allclose(Rac_Nusselt_Rayleigh_disk(4, 1, True), 51800)
    assert_allclose(Rac_Nusselt_Rayleigh_disk(H=1, D=.4, insulated=True), 51800)
    assert_allclose(Rac_Nusselt_Rayleigh_disk(H=1, D=.4, insulated=False), 151200)
    
    for r in (4,10, 100):
        assert_allclose(Rac_Nusselt_Rayleigh_disk(r, 1, False), 151200)
    
    
    for D in (5.9999999999, 6, 7, 50):
        assert_allclose(Rac_Nusselt_Rayleigh_disk(H=1, D=D, insulated=False), 1708.)
        assert_allclose(Rac_Nusselt_Rayleigh_disk(H=1, D=D, insulated=True), 1708.)

def test_Nu_Nusselt_Rayleigh_Hollands():
    assert_allclose(Nu_Nusselt_Rayleigh_Hollands(5.54, 3.21e8, buoyancy=True), 69.02668649510164)
    assert_allclose(Nu_Nusselt_Rayleigh_Hollands(.7, 3.21e6, buoyancy=True, Rac=Rac_Nusselt_Rayleigh(H=1, L=2, W=.2, insulated=False)), 4.666249131876477)
    
    assert_allclose(Nu_Nusselt_Rayleigh_Hollands(.7, 3.21e6, buoyancy=True, Rac=Rac_Nusselt_Rayleigh(H=1, L=1, W=1, insulated=False)), 8.786362614129537)
            
def test_Rac_Nusselt_Rayleigh_fit_uninsulated():
    from ht.conv_free_enclosed import tck_uninstulated_Catton, ratios_uninsulated_Catton, Racs_uninstulated_Catton
    all_zs = []
    all_xs = []
    all_ys = []
    for ratio1, Rac_row in zip(ratios_uninsulated_Catton, Racs_uninstulated_Catton):
        for Rac, ratio2 in zip(Rac_row, ratios_uninsulated_Catton):
            all_zs.append(Rac)
            all_xs.append(ratio1)
            all_ys.append(ratio2)
    
    tck = bisplrep(all_xs, all_ys, np.log(all_zs), kx=3, ky=3, s=0)
    
    for i in range(len(tck)):
        assert_allclose(tck[i], tck_uninstulated_Catton[i], rtol=1e-5)

#    for i, Racs in enumerate(Racs_uninstulated_Catton):
#        plt.semilogy(ratios_uninsulated_Catton, Racs, label=str(ratios_uninsulated_Catton[i]))
#        fit = np.exp(bisplev(ratios_uninsulated_Catton[i], ratios_uninsulated_Catton, tck))
#        plt.semilogy(ratios_uninsulated_Catton, fit, 'o')
#        
#    plt.legend()
#    plt.show()
    
def test_Rac_Nusselt_Rayleigh_fit_insulated():
    from ht.conv_free_enclosed import ratios_insulated_Catton, Racs_instulated_Catton, tck_insulated_Catton
    
    all_zs = []
    all_xs = []
    all_ys = []
    for ratio1, Rac_row in zip(ratios_insulated_Catton, Racs_instulated_Catton):
        for Rac, ratio2 in zip(Rac_row, ratios_insulated_Catton):
            if Rac is not None:
                all_zs.append(Rac)
                all_xs.append(ratio1)
                all_ys.append(ratio2)
    
    tck = bisplrep(all_xs, all_ys, np.log(all_zs), kx=1, ky=2, s=1e-4)
    for i in range(len(tck)):
        assert_allclose(tck[i], tck_insulated_Catton[i], rtol=1e-5)

#    for i, Racs in enumerate(Racs_instulated_Catton):
#        plt.semilogy(ratios_insulated_Catton, Racs, '-', label=str(ratios_insulated_Catton[i]))
#        fit = np.exp(bisplev(ratios_insulated_Catton[i], ratios_insulated_Catton, tck))
#        plt.semilogy(ratios_insulated_Catton, fit, 'o')
#        
#    plt.legend()
#    plt.show()


def test_Rac_Nusselt_Rayleigh_disk_fits():
    from fluids.optional import pychebfun
    from ht.conv_free_enclosed import insulated_disk_coeffs, uninsulated_disk_coeffs
    ratios = [0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 6]
    Ras_uninsulated = [151200, 66600, 21300, 8010, 4350, 2540, 2010, 1880, 1708]
    Ras_insulated = [51800, 23800, 8420, 3770, 2650, 2260, 1900, 1830, 1708]
    
    uninsulated = UnivariateSpline(ratios, 1/np.log(Ras_uninsulated), k=1, s=0)
    insulated = UnivariateSpline(ratios, 1/np.log(Ras_insulated), k=1, s=0)
    
    N = 8
    insulated_fun = pychebfun.chebfun(insulated, domain=[ratios[0], ratios[-1]], N=N)
    uninsulated_fun = pychebfun.chebfun(uninsulated, domain=[ratios[0], ratios[-1]], N=N)
    

    insulated_coeffs = pychebfun.chebfun_to_poly(insulated_fun)
    uninsulated_coeffs = pychebfun.chebfun_to_poly(uninsulated_fun)
    
    assert_allclose(insulated_coeffs, insulated_disk_coeffs)
    assert_allclose(uninsulated_coeffs, uninsulated_disk_coeffs)
    
#    more_ratios = np.logspace(np.log10(ratios[0]), np.log10(ratios[-1]), 1000)
#    plt.semilogy(ratios, Ras_insulated)
#    plt.semilogy(ratios, Ras_uninsulated)
#    
#    plt.semilogy(more_ratios, np.exp(1/insulated_fun(np.array(more_ratios))), 'x')
#    plt.semilogy(more_ratios, np.exp(1/uninsulated_fun(np.array(more_ratios))), 'o')
#    plt.show()



def test_Nu_vertical_helical_coil_Ali():
    Nu = Nu_vertical_helical_coil_Ali(4.4, 1E11)
    assert_allclose(Nu, 1808.5774997297106)