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

def mark_as_numba(func):
    func = pytest.mark.numba(func)
#    func = pytest.mark.slow(func)
    func = pytest.mark.skipif(numba is None, reason="Numba is missing")(func)
    return func

@mark_as_numba
def test_tube_bank():
    # not implemented - dP_Zukauskas

    kwargs = dict(Re=10263.37, Pr=.708, tube_rows=11, pitch_normal=.05, pitch_parallel=.05, Do=.025)
    assert_close(ht.numba.Nu_Grimison_tube_bank(**kwargs), ht.Nu_Grimison_tube_bank(**kwargs))

    assert_close(ht.numba.Zukauskas_tube_row_correction(4, staggered=True),
                 ht.Zukauskas_tube_row_correction(4, staggered=True))

    kwargs = dict(Re=1E4, Pr=7., tube_rows=10, pitch_parallel=.05, pitch_normal=.05)
    assert_close(ht.numba.Nu_Zukauskas_Bejan(**kwargs), ht.Nu_Zukauskas_Bejan(**kwargs))

    kwargs = dict(Re=1.32E4, Pr=0.71, tube_rows=8, pitch_parallel=.09, pitch_normal=.05)
    assert_close(ht.numba.Nu_ESDU_73031(**kwargs), ht.Nu_ESDU_73031(**kwargs))

    kwargs = dict(Re=10263.37, Pr=.708, tube_rows=11, pitch_normal=.05, pitch_parallel=.05, Do=.025)
    assert_close(ht.numba.Nu_HEDH_tube_bank(**kwargs), ht.Nu_HEDH_tube_bank(**kwargs))
    kwargs = dict(Re=10263.37, Pr=.708, tube_rows=5, pitch_normal=.05, pitch_parallel=.05, Do=.025)
    assert_close(ht.numba.Nu_HEDH_tube_bank(**kwargs), ht.Nu_HEDH_tube_bank(**kwargs))

    kwargs = dict(m=11., rho=995., mu=0.000803, mu_w=0.000657, DShell=0.584, LSpacing=0.1524, pitch=0.0254, Do=.019, NBaffles=22)
    assert_close(ht.numba.dP_Kern(**kwargs), ht.dP_Kern(**kwargs))

    assert_close(ht.numba.baffle_correction_Bell(0.82, 'Chebyshev'), ht.numba.baffle_correction_Bell(0.82, 'Chebyshev'))
    assert_close(ht.numba.baffle_correction_Bell(0.82), ht.numba.baffle_correction_Bell(0.82))

    assert_close(ht.numba.baffle_leakage_Bell(1, 3, 8), ht.baffle_leakage_Bell(1, 3, 8))
    assert_close(ht.numba.baffle_leakage_Bell(1, 3, 8, 'HEDH'), ht.baffle_leakage_Bell(1, 3, 8, 'HEDH'))

    assert_close(ht.numba.bundle_bypassing_Bell(0.5, 5, 25), ht.bundle_bypassing_Bell(0.5, 5, 25))
    assert_close(ht.numba.unequal_baffle_spacing_Bell(16, .1, .15, 0.15), ht.unequal_baffle_spacing_Bell(16, .1, .15, 0.15))


@mark_as_numba
def test_conv_internal():
    assert_close(ht.numba.Nu_conv_internal(Re=1E2, Pr=.7, x=.01, Di=.1),
                 ht.Nu_conv_internal(Re=1E2, Pr=.7, x=.01, Di=.1))


@mark_as_numba
def test_conv_free_immersed():
    assert_close(ht.numba.Nu_vertical_cylinder(0.72, 1E7, L=1.0, D=3.),
                 ht.Nu_vertical_cylinder(0.72, 1E7, L=1.0, D=3.0))


@mark_as_numba
def test_conv_free_enclosed():
    assert_close(ht.numba.Nu_Nusselt_Rayleigh_Holling_Herwig(5.54, 3.21e8, buoyancy=True),
                 ht.Nu_Nusselt_Rayleigh_Holling_Herwig(5.54, 3.21e8, buoyancy=True))

    assert_close(ht.numba.Rac_Nusselt_Rayleigh(1, .5, 2, False),
                 ht.Rac_Nusselt_Rayleigh(1, .5, 2, False))
    assert_close(ht.numba.Rac_Nusselt_Rayleigh(1, .5, 2, True),
                 ht.Rac_Nusselt_Rayleigh(1, .5, 2, True))

    assert_close(ht.numba.Rac_Nusselt_Rayleigh_disk(H=1, D=4, insulated=False),
                 ht.Rac_Nusselt_Rayleigh_disk(H=1, D=4, insulated=False))
    assert_close(ht.numba.Rac_Nusselt_Rayleigh_disk(H=1, D=4, insulated=True),
                 ht.Rac_Nusselt_Rayleigh_disk(H=1, D=4, insulated=True))

    assert_close(ht.numba.Nu_free_vertical_plate(0.69, 2.63E9, False),
                 ht.Nu_free_vertical_plate(0.69, 2.63E9, False))

    assert_close(ht.numba.Nu_free_horizontal_plate(5.54, 3.21e8, buoyancy=True),
                 ht.Nu_free_horizontal_plate(5.54, 3.21e8, buoyancy=True))




@mark_as_numba
def test_conv_external():
    assert_close(ht.numba.Nu_cylinder_Whitaker(6071.0, 0.7), ht.Nu_cylinder_Whitaker(6071.0, 0.7))
    assert_close(ht.numba.Nu_cylinder_Perkins_Leppert_1962(6071.0, 0.7),
                 ht.Nu_cylinder_Perkins_Leppert_1962(6071.0, 0.7))
    assert_close(ht.numba.Nu_cylinder_Perkins_Leppert_1964(6071.0, 0.7),
                 ht.Nu_cylinder_Perkins_Leppert_1964(6071.0, 0.7))

    assert ht.numba.Nu_external_cylinder_methods(0.72, 1E7) == ht.Nu_external_cylinder_methods(0.72, 1E7)

    assert_close(ht.numba.Nu_external_cylinder(6071, 0.7), ht.Nu_external_cylinder(6071, 0.7))

    assert Nu_external_horizontal_plate_methods(Re=1e7, Pr=.7) == ht.numba.Nu_external_horizontal_plate_methods(Re=1e7, Pr=.7)

    assert_close(ht.numba.Nu_external_horizontal_plate(Re=1E7, Pr=.7),
                 ht.Nu_external_horizontal_plate(Re=1E7, Pr=.7))


@mark_as_numba
def test_core_misc():
    assert_close(ht.numba.LMTD(100., 60., 20., 60, counterflow=False),
                 ht.LMTD(100., 60., 20., 60, counterflow=False))

    assert_close(ht.numba.fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58),
                 ht.fin_efficiency_Kern_Kraus(0.0254, 0.05715, 3.8E-4, 200, 58),)

    assert_close(ht.numba.wall_factor(mu=8E-4, mu_wall=3E-4, Pr=1.2, Pr_wall=1.1, T=300,T_wall=350, property_option='Prandtl'),
                 ht.wall_factor(mu=8E-4, mu_wall=3E-4, Pr=1.2, Pr_wall=1.1, T=300,T_wall=350, property_option='Prandtl'))


@mark_as_numba
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

    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=36*foot,
    tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
    angle=30, pitch_normal=2.5*inch, fin_height=0.625*inch, corbels=True)

    kwargs = dict(m=130.70315, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
        A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
        fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
        fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
        pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal,
        rho=1.2013848, Cp=1009.0188, mu=1.9304793e-05, k=0.027864828, k_fin=238)
    h_numba = ht.numba.h_Ganguli_VDI(**kwargs)
    h_normal = h_Ganguli_VDI(**kwargs)
    assert_close(h_numba, h_normal, rtol=1e-11)




@mark_as_numba
def test_boiling_flow():
    assert_close(ht.numba.Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, q=1E5),
                 ht.Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, q=1E5))
    Te = 32.04944566414243
    assert_close(ht.numba.Thome(m=10, x=0.5, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, Te=Te),
                       ht.Thome(m=10, x=0.5, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2, mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5, Psat=1E5, Pc=22E6, Te=Te))

    assert_close(ht.numba.Lazarek_Black(m=10, D=0.3, mul=1E-3, kl=0.6, Hvap=2E6, Te=100),
                 ht.Lazarek_Black(m=10, D=0.3, mul=1E-3, kl=0.6, Hvap=2E6, Te=100))

    assert_close(ht.numba.Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, q=1E5),
                 ht.Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, q=1E5))

    kwargs = dict(m=1.0, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, Te=10.0)
    assert_close(ht.numba.Sun_Mishima(**kwargs), ht.Sun_Mishima(**kwargs))

    kwargs = dict(m=1.0, x=0.4, D=0.3, rhol=567., mul=156E-6, sigma=0.02, Hvap=9E5, q=1E4)
    assert_close(ht.numba.Yun_Heo_Kim(**kwargs), ht.Yun_Heo_Kim(**kwargs))

    kwargs = dict(m=0.106, x=0.2, D=0.0212, rhol=567, rhog=18.09, mul=156E-6, mug=7.11E-6, kl=0.086, Cpl=2730, Hvap=2E5, sigma=0.02, dPsat=1E5, Te=3)
    assert_close(ht.numba.Chen_Edelstein(**kwargs), ht.Chen_Edelstein(**kwargs))

    kwargs = dict(m=0.106, x=0.2, D=0.0212, rhol=567.0, rhog=18.09, mul=156E-6, mug=7.11E-6, kl=0.086, Cpl=2730.0, Hvap=2E5, sigma=0.02, dPsat=1E5, Te=3.0)
    assert_close(ht.numba.Chen_Bennett(**kwargs), ht.Chen_Bennett(**kwargs))

    kwargs = dict(m=1.0, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, Cpl=2300.0, P=1E6, Pc=22E6, MW=44.02, Te=7.0)
    assert_close(ht.numba.Liu_Winterton(**kwargs), ht.Liu_Winterton(**kwargs))


@mark_as_numba
def test_radiation():
    assert_close(ht.numba.radiation. blackbody_spectral_radiance(800., 4E-6),
                 ht.radiation. blackbody_spectral_radiance(800., 4E-6))


@mark_as_numba
def test_conv_jacket():
    assert_close(ht.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8),
                 ht.numba.Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8))

    assert_close(ht.numba.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6),
                 ht.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6))

    assert_close(ht.numba.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303),
                 ht.Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20., rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, inlettype='radial', isobaric_expansion=0.000303))


@mark_as_numba
def test_condensation():
    assert_close(ht.numba.Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900, L=0.1),
                 ht.Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091, mul=158.9E-6, Hvap=776900, L=0.1))

    assert_close(ht.numba.Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85),
                 ht.Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098,  mul=159E-6, Cpl=2520., D=0.03, x=0.85))

    assert_close(ht.numba.Cavallini_Smith_Zecchin(m=1, x=0.4, D=.3, rhol=800, rhog=2.5, mul=1E-5, mug=1E-3, kl=0.6, Cpl=2300),
                 ht.Cavallini_Smith_Zecchin(m=1, x=0.4, D=.3, rhol=800, rhog=2.5, mul=1E-5, mug=1E-3, kl=0.6, Cpl=2300))


    assert_close(ht.numba.Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7),
                 ht.Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7))



@mark_as_numba
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



@mark_as_numba
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


@mark_as_numba
def test_boiling_nucleic():
    assert_close(ht.numba.Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26),
                 ht.Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26))

    numba_methods = ht.numba.h_nucleic_methods(P=3E5, Pc=22048320., Te=4.0, CAS='7732-18-5')
    regular_methods = ht.h_nucleic_methods(P=3E5, Pc=22048320., Te=4.0, CAS='7732-18-5')
    assert numba_methods == regular_methods


    # Has a TON of arguments, and numba wants them all to not be Nones.
    assert_close(ht.numba.h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26, P=1e4, Pc=1e6, Tsat=10, MW=33.0, Method='Rohsenow'),
                 ht.h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217, Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26, P=1e4, Pc=1e6, Tsat=10, MW=33.0, Method='Rohsenow'))

    kwargs = dict(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567.0, rhog=18.09, P=1e6, Pc=1e7)
    assert_close(ht.numba.qmax_boiling(**kwargs), ht.qmax_boiling(**kwargs))

    kwargs = dict(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    assert ht.qmax_boiling_methods(**kwargs) == ht.numba.qmax_boiling_methods(**kwargs)



@mark_as_numba
def test_packed_bed():
    # All good
    assert_close(ht.numba.Nu_packed_bed_Gnielinski(dp=8E-4, voidage=0.4, vs=1, rho=1E3, mu=1E-3, Pr=0.7),
                 ht.Nu_packed_bed_Gnielinski(dp=8E-4, voidage=0.4, vs=1, rho=1E3, mu=1E-3, Pr=0.7))



@mark_as_numba
def test_two_phase():
    kwargs = dict(m=1.0, x=.9, D=.3, alpha=.9, rhol=1000.0, rhog=2.2, mug=1e-5, Cpl=2300.0, kl=.6, mul=1e-3, mu_b=1E-3, mu_w=1.2E-3, L=5)
    assert_close(ht.numba.h_two_phase(**kwargs), ht.h_two_phase(**kwargs))


@mark_as_numba
def test_supercritical():
    assert_close(ht.numba.Nu_Griem(1E5, 1.2), ht.Nu_Griem(1E5, 1.2))

    kwargs = dict(Re=1E5, Pr=1.2, rho_w=125.8, rho_b=249.0233,
              Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700, T_pc=300.0)
    assert_close(ht.numba.Nu_Jackson(**kwargs), ht.Nu_Jackson(**kwargs))
    assert_close(ht.numba.Nu_Jackson(1E5, 1.2), ht.Nu_Jackson(1E5, 1.2))

    assert_close(ht.numba.Nu_Gupta(1E5, 1.2), ht.Nu_Gupta(1E5, 1.2))
    assert_close(ht.numba.Nu_Swenson(1E5, 1.2), ht.Nu_Swenson(1E5, 1.2))
    assert_close(ht.numba.Nu_Xu(1E5, 1.2), ht.Nu_Xu(1E5, 1.2))

    assert_close(ht.numba.Nu_Mokry(1E5, 1.2), ht.Nu_Mokry(1E5, 1.2))
    assert_close(ht.numba.Nu_Ornatsky(1E5, 1.2, 1.5), ht.Nu_Ornatsky(1E5, 1.2, 1.5))
    assert_close(ht.numba.Nu_Zhu(1E5, 1.2), ht.Nu_Zhu(1E5, 1.2))

    kwargs = dict(Re=1E5, Pr=1.2, Pr_pc=1.5, Cp_avg=2080.845, Cp_b=2048.621, T_b=650, T_w=700, T_pc=600.0)
    assert_close(ht.numba.Nu_Yamagata(**kwargs), ht.Nu_Yamagata(**kwargs))

    assert_close(ht.numba.Nu_Kitoh(1E5, 1.2), ht.Nu_Kitoh(1E5, 1.2))

    assert_close(ht.numba.Nu_Krasnoshchekov_Protopopov(1E5, 1.2),
                 ht.Nu_Krasnoshchekov_Protopopov(1E5, 1.2))

    assert_close(ht.numba.Nu_Petukhov(1E5, 1.2),
                 ht.Nu_Petukhov(1E5, 1.2))

    assert_close(ht.numba.Nu_Krasnoshchekov(1E5, 1.2),
                 ht.Nu_Krasnoshchekov(1E5, 1.2))




@mark_as_numba
def test_conduction():
    assert_close(ht.numba.R_value_to_k(1., SI=False), ht.numba.R_value_to_k(1., SI=False))

    assert_close(ht.numba.S_isothermal_pipe_to_isothermal_pipe(.1, .2, 1, 1),
                 ht.S_isothermal_pipe_to_isothermal_pipe(.1, .2, 1, 1))

    # cylindrical_heat_transfer returns a dictionary, not supported by numba

@mark_as_numba
def test_hx_tube_bundles():
    kwargs = dict(Ntubes=782, Do=.028, pitch=.036, Ntp=2, angle=45.)
    assert_close(ht.numba.DBundle_for_Ntubes_Phadkeb(**kwargs),
                 ht.DBundle_for_Ntubes_Phadkeb(**kwargs))

    kwargs = dict(DBundle=1.184, Do=.028, Ntp=2, angle=45)
    assert_close(ht.numba.Ntubes_Perrys(**kwargs), ht.Ntubes_Perrys(**kwargs))

    assert_close(ht.numba.Ntubes_VDI(DBundle=1.184, Ntp=2, Do=.028, pitch=.036, angle=30),
                 ht.Ntubes_VDI(DBundle=1.184, Ntp=2, Do=.028, pitch=.036, angle=30) )

    kwargs = dict(DBundle=1.2, Do=0.025, pitch=0.03125, Method='Phadkeb')
    assert ht.numba.Ntubes(**kwargs) == ht.Ntubes(**kwargs)

    kwargs = dict(N=1285, Do=0.025, pitch=0.03125)
    assert_close(ht.numba.size_bundle_from_tubecount(**kwargs),
                 ht.size_bundle_from_tubecount(**kwargs))

@mark_as_numba
def test_hx_data():
    assert_close(ht.L_unsupported_max(Do=.0254, material='CS'),
                 ht.numba.L_unsupported_max(Do=.0254, material='CS'))

    assert_close(ht.numba.baffle_thickness(Dshell=.3, L_unsupported=50, service='R'),
                 ht.baffle_thickness(Dshell=.3, L_unsupported=50, service='R'))

@mark_as_numba
def test_hx_effectiveness_basic():
    R1 = 3.811315897216142e-05
    NTU1 = 0.31156549511556475
    assert_close(ht.numba.temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow'),
                 ht.temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow'))

    P1 = ht.numba.temperature_effectiveness_basic(R1=R1, NTU1=NTU1,
                                    subtype='crossflow')
    NTU1_calc = ht.numba.NTU_from_P_basic(P1, R1, subtype='crossflow')
    assert_close(NTU1, NTU1_calc)

@mark_as_numba
def test_hx_effectiveness():
    assert_close(ht.numba.temperature_effectiveness_air_cooler(.5, 2, rows=10, passes=10),
                 ht.temperature_effectiveness_air_cooler(.5, 2, rows=10, passes=10))

    assert_close(ht.numba.hx.Pp(5, .4),  ht.hx.Pp(5, .4))
    assert_close(ht.numba.hx.Pc(5, .4),  ht.hx.Pc(5, .4))

    # Quite literally 20x faster than CPython; Pypy is unfortunately slow as there is a scipy function

    assert_close(ht.numba.NTU_from_P_J(P1=.99, R1=.01, Ntp=2),
                 ht.NTU_from_P_J(P1=.99, R1=.01, Ntp=2))

    assert_close(ht.numba.NTU_from_P_G(P1=.573, R1=1/3., Ntp=1),
                 ht.NTU_from_P_G(P1=.573, R1=1/3., Ntp=1))

    assert_close(ht.numba.NTU_from_P_E(P1=.58, R1=1/3., Ntp=2),
                 ht.NTU_from_P_E(P1=.58, R1=1/3., Ntp=2))

    assert_close(ht.numba.NTU_from_P_H(P1=0.573, R1=1/3., Ntp=1),
                 ht.NTU_from_P_H(P1=0.573, R1=1/3., Ntp=1))

    assert_close(ht.numba.NTU_from_P_plate(P1=0.5743, R1=1/3., Np1=3, Np2=1),
                 ht.NTU_from_P_plate(P1=0.5743, R1=1/3., Np1=3, Np2=1))


@mark_as_numba
def test_conv_plate():
    kwargs = dict(Re=2000, Pr=0.7, chevron_angle=30, mu=1E-3, mu_wall=8E-4)
    assert_close(ht.numba.Nu_plate_Kumar(**kwargs), ht.Nu_plate_Kumar(**kwargs))

    kwargs = dict(Re=2000, Pr=.7, plate_enlargement_factor=1.18)
    assert_close(ht.numba.Nu_plate_Martin(**kwargs), ht.Nu_plate_Martin(**kwargs))