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
def test_radiation():
    assert_close(ht.numba.radiation. blackbody_spectral_radiance(800., 4E-6),
                 ht.radiation. blackbody_spectral_radiance(800., 4E-6))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_conv_jacket():
    assert_close(ht.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8),
                 ht.numba.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8))

    assert_close(ht.numba.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6),
                 ht.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6))
    
    assert_close(ht.numba.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303),
                 ht.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_condensation():
    assert_close(ht.numba.Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900, L=0.1),
                 ht.Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900, L=0.1))
    
    assert_close(ht.numba.Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85),
                 ht.Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85))
    
    assert_close(ht.numba.Cavallini_Smith_Zecchin(m=1, x=0.4, D=.3, rhol=800, rhog=2.5, mul=1E-5, mug=1E-3, kl=0.6, Cpl=2300),
                 ht.Cavallini_Smith_Zecchin(m=1, x=0.4, D=.3, rhol=800, rhog=2.5, mul=1E-5, mug=1E-3, kl=0.6, Cpl=2300))
    
    
    assert_close(ht.numba.Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7),
                 ht.Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7))


@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_boiling_plate():
    assert_close(ht.numba.h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.00172, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, A_channel_flow=0.0003),
                 ht.h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.00172, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, A_channel_flow=0.0003))
    
    assert_close(ht.numba.h_boiling_Lee_Kang_Kim(m=3E-5, x=.4, D_eq=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=9E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003),
                 ht.h_boiling_Lee_Kang_Kim(m=3E-5, x=.4, D_eq=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, mug=9E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003))
    
    assert_close(ht.numba.h_boiling_Han_Lee_Kim(m=3E-5, x=.4, Dh=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6,  Hvap=9E5, Cpl=2200, q=1E5, A_channel_flow=0.0003, wavelength=3.7E-3, chevron_angle=45),
                 ht.h_boiling_Han_Lee_Kim(m=3E-5, x=.4, Dh=0.002, rhol=567., rhog=18.09, kl=0.086, mul=156E-6,  Hvap=9E5, Cpl=2200, q=1E5, A_channel_flow=0.0003, wavelength=3.7E-3, chevron_angle=45))
    
    assert_close(ht.numba.h_boiling_Huang_Sheer(rhol=567., rhog=18.09, kl=0.086, mul=156E-6, Hvap=9E5, sigma=0.02, Cpl=2200, q=1E4, Tsat=279.15),
                 ht.h_boiling_Huang_Sheer(rhol=567., rhog=18.09, kl=0.086, mul=156E-6, Hvap=9E5, sigma=0.02, Cpl=2200, q=1E4, Tsat=279.15))
    
    assert_close(ht.numba.h_boiling_Yan_Lin(m=3E-5, x=.4, Dh=0.002, rhol=567., rhog=18.09, kl=0.086, Cpl=2200, mul=156E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003),
                 ht.h_boiling_Yan_Lin(m=3E-5, x=.4, Dh=0.002, rhol=567., rhog=18.09, kl=0.086, Cpl=2200, mul=156E-6, Hvap=9E5, q=1E5, A_channel_flow=0.0003))


@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_Ntubes_Phadkeb():
    # Extremely impressive performance
    Bundles = np.linspace(1, 2, 5)
    Dos = np.linspace(.028, .029, 5)
    pitches = np.linspace(.036, .037, 5)
    Ntps = np.linspace(2, 2, 5, dtype=np.int64)
    angles = np.linspace(45, 45, 5, dtype=np.int64)
    
    assert 782 == ht.numba.Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=2, angle=45.)
    
    
    
#    assert_close(ht.numba_vectorized.Ntubes_Phadkeb(Bundles, Dos, pitches, Ntps, angles), 
#                 [ 558,  862, 1252, 1700, 2196])
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_boiling_nucleic():
    assert_close(ht.numba.Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26),
                 ht.Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26))

    numba_methods = ht.numba.h_nucleic_methods(P=3E5, Pc=22048320., Te=4.0, CAS='7732-18-5')
    regular_methods = ht.h_nucleic_methods(P=3E5, Pc=22048320., Te=4.0, CAS='7732-18-5')
    assert numba_methods == regular_methods
    

    # Has a TON of arguments, and numba wants them all to not be Nones.
    assert_close(ht.numba.h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26, P=1e4, Pc=1e6, Tsat=10, MW=33.0, Method='Rohsenow'),
                 ht.h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26, P=1e4, Pc=1e6, Tsat=10, MW=33.0, Method='Rohsenow'))

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_packed_bed():
    # All good
    assert_close(ht.numba.Nu_packed_bed_Gnielinski(dp=8E-4, voidage=0.4, vs=1, rho=1E3, mu=1E-3, Pr=0.7),
                 ht.Nu_packed_bed_Gnielinski(dp=8E-4, voidage=0.4, vs=1, rho=1E3, mu=1E-3, Pr=0.7))
    
@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_conduction():
    assert_close(ht.numba.R_value_to_k(1., SI=False), ht.numba.R_value_to_k(1., SI=False))
    
    assert_close(ht.numba.S_isothermal_pipe_to_isothermal_pipe(.1, .2, 1, 1),
                 ht.S_isothermal_pipe_to_isothermal_pipe(.1, .2, 1, 1))

    # cylindrical_heat_transfer returns a dictionary, not supported by numba

@pytest.mark.numba
@pytest.mark.skipif(numba is None, reason="Numba is missing")
def test_hx():
    assert_close(ht.numba.temperature_effectiveness_air_cooler(.5, 2, rows=10, passes=10),
                 ht.temperature_effectiveness_air_cooler(.5, 2, rows=10, passes=10))
