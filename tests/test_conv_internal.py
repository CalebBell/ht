# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018, 2019, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from ht.boiling_nucleic import _angles_Stephan_Abdelsalam
import numpy as np

from numpy.testing import assert_allclose
import pytest


### conv_internal
def test_Nu_const():
    assert_allclose(laminar_T_const(), 3.66)
    assert_allclose(laminar_Q_const(), 48/11.)


def test_laminar_entry_region():
    Nu = laminar_entry_thermal_Hausen(100000, 1.1, 5, .5)
    assert_allclose(Nu, 39.01352358988535)

    Nu = laminar_entry_Seider_Tate(Re=100000, Pr=1.1, L=5, Di=.5)
    assert_allclose(Nu, 41.366029684589265)
    Nu_wall = laminar_entry_Seider_Tate(100000, 1.1, 5, .5, 1E-3, 1.2E-3)
    assert_allclose(Nu_wall, 40.32352264095969)

    Nu = laminar_entry_Baehr_Stephan(100000, 1.1, 5, .5)
    assert_allclose(Nu, 72.65402046550976)

def test_turbulent_complicated():
    Nu1 = turbulent_Dittus_Boelter(1E5, 1.2, True, False)
    Nu2 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False, revised=False)
    Nu3 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False)
    Nu4 = turbulent_Dittus_Boelter(Re=1E5, Pr=1.2)
    Nu_values = [261.3838629346147, 279.89829163640354, 242.9305927410295, 247.40036409449127]
    assert_allclose([Nu1, Nu2, Nu3, Nu4], Nu_values)

    Nu1 = turbulent_Sieder_Tate(Re=1E5, Pr=1.2)
    Nu2 = turbulent_Sieder_Tate(1E5, 1.2, 0.01, 0.067)
    assert_allclose([Nu1, Nu2], [286.9178136793052, 219.84016455766044])

    Nus = [turbulent_entry_Hausen(1E5, 1.2, 0.154, i) for i in np.linspace(0,1,11)]
    Nus_values = [np.inf, 507.39810608575436, 400.1002551153033, 356.83464396632377, 332.50684459222612, 316.60088883614151, 305.25121748064328, 296.67481510644825, 289.92566421612082, 284.45128111774227, 279.90553997822707]
    assert_allclose(Nus, Nus_values)

def test_turbulent_simple():
    Nu = turbulent_Colburn(1E5, 1.2)
    assert_allclose(Nu, 244.41147091200068)

    Nu = turbulent_Drexel_McAdams(1E5, 0.6)
    assert_allclose(Nu, 171.19055301724387)

    Nu = turbulent_von_Karman(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 255.7243541243272)

    Nu = turbulent_Prandtl(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 256.073339689557)

    Nu = turbulent_Friend_Metzner(1E5, 100., 0.0185)
    assert_allclose(Nu, 1738.3356262055322)

    Nu = turbulent_Petukhov_Kirillov_Popov(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 250.11935088905105)

    Nu = turbulent_Webb(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 239.10130376815872)

    Nu = turbulent_Sandall(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 229.0514352970239)

    Nu = turbulent_Gnielinski(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 254.62682749359632)

    Nu = turbulent_Gnielinski_smooth_1(1E5, 1.2)
    assert_allclose(Nu, 227.88800494373442)

    Nu = turbulent_Gnielinski_smooth_2(1E5, 7.)
    assert_allclose(Nu, 577.7692524513449)

    Nu = turbulent_Churchill_Zajic(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 260.5564907817961)

    Nu = turbulent_ESDU(1E5, 1.2)
    assert_allclose(Nu, 232.3017143430645)

def test_turbulent_rough():
    Nu = turbulent_Martinelli(1E5, 100., 0.0185)
    assert_allclose(Nu, 887.1710686396347)

    Nu = turbulent_Nunner(1E5, 0.7, 0.0185, 0.005)
    assert_allclose(Nu, 101.15841010919947)

    Nu = turbulent_Dipprey_Sabersky(1E5, 1.2, 0.0185, 1E-3)
    assert_allclose(Nu, 288.33365198566656)

    Nu = turbulent_Gowen_Smith(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 131.72530453824106)

    Nu = turbulent_Kawase_Ulbrecht(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 389.6262247333975)

    Nu = turbulent_Kawase_De(1E5, 1.2, 0.0185)
    assert_allclose(Nu, 296.5019733271324)

    Nu = turbulent_Bhatti_Shah(1E5, 1.2, 0.0185, 1E-3)
    assert_allclose(Nu, 302.7037617414273)

# TODO meta function  Nu internal


def test_Morimoto_Hotta():
    Nu = Morimoto_Hotta(1E5, 5.7, .05, .5)
    assert_allclose(Nu, 634.4879473869859)
    
    
def test_helical_turbulent_Nu_Mori_Nakayama():
    Nu = helical_turbulent_Nu_Mori_Nakayama(2E5, 0.7, 0.01, .2)
    assert_allclose(Nu, 496.2522480663327)
    # High Pr
    Nu = helical_turbulent_Nu_Mori_Nakayama(2E5, 4, 0.01, .2)
    assert_allclose(Nu, 889.3060078437253)
    
    # Bad behavior!
    # 1 sun power output per m^2 per K
    assert 4E24 < helical_turbulent_Nu_Mori_Nakayama(2E6, 0.7, 1, 1E80)
    
    # .[3]_ specified that the high-Pr formula is calculated using Dean number, 
    # but the actual article says it is not. We use the 2.5 power specified 
    # in the original.
    
def test_helical_turbulent_Nu_Schmidt():
    Nu = helical_turbulent_Nu_Schmidt(2E5, 0.7, 0.01, .2)
    assert_allclose(Nu, 466.2569996832083)
    Nus = [helical_turbulent_Nu_Schmidt(i, 0.7, 0.01, .2) for i in [2.2E4, 2.2E4+1E-9]]
    assert_allclose(Nus, [80.1111786843, 79.75161984693375])
    
    
def test_helical_turbulent_Nu_Xin_Ebadian():
    Nu = helical_turbulent_Nu_Xin_Ebadian(2E5, 0.7, 0.01, .2)
    assert_allclose(Nu, 474.11413424344755)
    
    # No bad behavior
    # Checked with the original
    
def test_Nu_laminar_rectangular_Shan_London():
    Nu = Nu_laminar_rectangular_Shan_London(.7)
    assert_allclose(Nu, 3.751762675455)
    
def test_Nu_conv_internal_methods():
    from fluids.units import func_args
    
    for name, (func, args) in conv_tube_methods.items():
        assert tuple(list(func_args(func))[0:len(args)]) == args
    

def test_Nu_conv_internal():
    Nu = Nu_conv_internal(1E2, .7)
    assert_allclose(Nu, laminar_T_const())
    
    Nu = Nu_conv_internal(1E2, .7, Method='Laminar - constant Q')
    assert_allclose(Nu, laminar_Q_const())

    Nu = Nu_conv_internal(1E2, .7, x=.01, Di=.1)
    assert_allclose(Nu, 14.91799128769779)
    
    # test the other laminar entrylength methods
    Nu = Nu_conv_internal(1E2, .7, x=.01, Di=.1, Method='Hausen laminar thermal entry')
    assert_allclose(Nu, 16.51501443241237)
    Nu = Nu_conv_internal(1E2, .7, x=.01, Di=.1, Method='Seider-Tate laminar thermal entry')
    assert_allclose(Nu, 21.054212255270848)
    
    # martinili
    Nu = Nu_conv_internal(1E5, .02, eD=0)
    assert_allclose(Nu, 8.246171632616187)
    
    Nu = Nu_conv_internal(1E5, .7, x=.01, Di=.1)
    assert_allclose(Nu, 978.1729258857774)

    Nu = Nu_conv_internal(1E5, .7)
    assert_allclose(Nu, 183.71057902604906)
    
    other_methods = ['Churchill-Zajic', 'Petukhov-Kirillov-Popov', 'Gnielinski',
                     'Bhatti-Shah', 'Dipprey-Sabersky', 'Sandall', 'Webb',
                     'Friend-Metzner', 'Prandtl', 'von-Karman', 'Gowen-Smith', 
                     'Kawase-Ulbrecht', 'Kawase-De', 'Nunner', 'Dittus-Boelter', 
                     'Sieder-Tate', 'Drexel-McAdams', 'Colburn', 'ESDU', 
                     'Gnielinski smooth low Pr','Gnielinski smooth high Pr']
    
    expected = [103.65851760127596, 96.66083769419261, 95.7206648591076, 
                124.96666518189072, 124.96666518189072, 126.8559349821517,
                89.04183860378171, 82.62190521404274, 96.39509181385534,
                97.64409839390211, 63.69345925482798, 218.78659693866075,
                169.9758751276217, 113.72592148878971, 199.41923780765848, 
                239.73408047050233, 182.078434520036, 204.21792040079825, 
                177.52639370276017, 183.6911292257849, 230.01408232621412]
    
    
    for method, expect in zip(other_methods, expected):
        Nu = Nu_conv_internal(1E5, .7, fd=.01, Method=method)
        assert_allclose(Nu, expect)
        
    with pytest.raises(Exception):
        Nu_conv_internal(1E5, .7, Method='NOTAMETHOD')

    l = Nu_conv_internal(1E5, .7, AvailableMethods=True)
    assert len(l) == 21
        
test_Nu_conv_internal()