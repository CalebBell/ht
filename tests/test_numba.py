# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids import *
from ht import *
import ht.vectorized
from math import *
from fluids.constants import *
from fluids.numerics import assert_close, assert_close1d
import pytest
try:
    import numba
    import ht.numba
    import ht.numba_vectorized
except:
    numba = None
import numpy as np

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_core_misc():
    assert_close(ht.numba.LMTD(100., 60., 20., 60, counterflow=False),
                 ht.LMTD(100., 60., 20., 60, counterflow=False))
    
    assert_close(ht.numba.fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58),
                 ht.fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58),)

    assert_close(ht.numba.wall_factor(mu=8E-4, mu_wall=3E-4, Pr=1.2, Pr_wall=1.1, T=300,T_wall=350, property_option='Prandtl'),
                 ht.wall_factor(mu=8E-4, mu_wall=3E-4, Pr=1.2, Pr_wall=1.1, T=300,T_wall=350, property_option='Prandtl'))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_air_cooler():
    assert_close(ht.numba.Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4),
                 ht.air_cooler.Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4))

    assert_close(ht.numba.air_cooler_noise_GPSA(tip_speed=3177/60, power=25.1*750),
                 ht.air_cooler_noise_GPSA(tip_speed=3177/60, power=25.1*750))


    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=20, tube_length=3, 
    tube_diameter=1*inch, fin_thickness=0.000406, fin_density=1/0.002309,
    pitch_normal=.06033, pitch_parallel=.05207,
    fin_height=0.0159, tube_thickness=(.0254-.0186)/2,
    bundles_per_bay=1, parallel_bays=1, corbels=True)
    
    # 2.5 us numba, 30 us CPython, 100 us PyPy
    assert_close(ht.numba.h_Briggs_Young(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin, A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter, fin_diameter=AC.fin_diameter, bare_length=AC.bare_length, fin_thickness=AC.fin_thickness, rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205),
                ht.h_Briggs_Young(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin, A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter, fin_diameter=AC.fin_diameter, bare_length=AC.bare_length, fin_thickness=AC.fin_thickness, rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205),)
    
    assert_close(ht.numba.h_ESDU_high_fin(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin, A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter, fin_diameter=AC.fin_diameter, bare_length=AC.bare_length, fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows, pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205, Pr_wall=7.0),
                ht.h_ESDU_high_fin(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin, A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter, fin_diameter=AC.fin_diameter, bare_length=AC.bare_length, fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows, pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205, Pr_wall=7.0))

    assert_close(ht.numba.h_ESDU_low_fin(m=0.914, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin, A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter, fin_diameter=AC.fin_diameter, bare_length=AC.bare_length, fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows, pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, rho=1.217, Cp=1007., mu=1.8E-5, k=0.0253, k_fin=15),
                 ht.h_ESDU_low_fin(m=0.914, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin, A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter, fin_diameter=AC.fin_diameter, bare_length=AC.bare_length, fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows, pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, rho=1.217, Cp=1007., mu=1.8E-5, k=0.0253, k_fin=15))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_flow_boiling():
    assert_close(ht.numba.Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, q=1E5),
                 ht.Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, q=1E5))
    Te = 32.04944566414243
    assert_close(ht.numba.Thome(m=10, x=0.5, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, Te=Te),
                       ht.Thome(m=10, x=0.5, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, Te=Te))



@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_conv_tank():
    assert_close(ht.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8),
                 ht.numba.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8))



@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Ntubes_Phadkeb():
    # Extremely impressive performance
    Bundles = np.linspace(1, 2, 5)
    Dos = np.linspace(.028, .029, 5)
    pitches = np.linspace(.036, .037, 5)
    Ntps = np.linspace(2, 2, 5, dtype=np.int64)
    angles = np.linspace(45, 45, 5, dtype=np.int64)
    
    assert_close(ht.numba_vectorized.Ntubes_Phadkeb(Bundles, Dos, pitches, Ntps, angles), 
                 [ 558,  862, 1252, 1700, 2196])
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_boiling_nucleic():
    assert_close(ht.numba.Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26),
                 ht.Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26))
