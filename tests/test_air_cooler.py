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
from random import choice
from fluids.geometry import *
from scipy.constants import minute, hp, inch, foot
from ht.boiling_nucleic import _angles_Stephan_Abdelsalam
from numpy.testing import assert_allclose
import pytest

### Air Cooler

def test_air_cooler_Ft():    
    Ft_1 = Ft_aircooler(Thi=93, Tho=52, Tci=35, Tco=54.59, Ntp=2, rows=4)
    assert_allclose(Ft_1, 0.9570456123827129)
    
    # Example 2 as in [1]_; author rounds to obtain a slightly different result.
    Ft_2 = Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)
    assert_allclose(Ft_2, 0.5505093604092708)
    Ft_many = [[Ft_aircooler(Thi=125., Tho=80., Tci=25., Tco=95., Ntp=i, rows=j) for i in range(1,6)] for j in range(1, 6)]
    Ft_values = [[0.6349871996666123, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244], [0.7993839562360742, 0.9184594715750571, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244], [0.8201055328279105, 0.9392743008890244, 0.9784008071402877, 0.9392743008890244, 0.9392743008890244], [0.8276966706732202, 0.9392743008890244, 0.9392743008890244, 0.9828365967034366, 0.9392743008890244], [0.8276966706732202, 0.9392743008890244, 0.9392743008890244, 0.9392743008890244, 0.9828365967034366]]
    assert_allclose(Ft_many, Ft_values)
    
    
def test_air_cooler_noise_GPSA():
    noise = air_cooler_noise_GPSA(tip_speed=3177/minute, power=25.1*hp)
    assert_allclose(noise, 100.53680477959792)
    
    
def test_air_cooler_noise_Mukherjee():
    '''    # Confirmed to be log10's because of example tip speed reduction
    # of 60 m/s to 40 m/s saves 5.3 dB.
    # hp in shaft horse power
    # d in meters
    # sound pressure level, ref level 2E-5 pa

    '''
    noise = air_cooler_noise_Mukherjee(tip_speed=3177/minute, power=25.1*hp, fan_diameter=4.267)
    assert_allclose(noise, 99.11026329092925)
    
    noise = air_cooler_noise_Mukherjee(tip_speed=3177/minute, power=25.1*hp, fan_diameter=4.267, induced=True)
    assert_allclose(noise, 96.11026329092925)
    
    
def test_h_ESDU_high_fin():

    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=20, tube_length=3, 
                            tube_diameter=1*inch, fin_thickness=0.000406, fin_density=1/0.002309,
                            pitch_normal=.06033, pitch_parallel=.05207,
                            fin_height=0.0159, tube_thickness=(.0254-.0186)/2,
                            bundles_per_bay=1, parallel_bays=1, corbels=True)
    h_bare_tube_basis = h_ESDU_high_fin(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
                         A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
                         fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
                         fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
                         pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, 
                         rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205)
    assert_allclose(h_bare_tube_basis, 1390.888918049757)


def test_h_ESDU_low_fin():
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=8, tube_length=0.5, 
    tube_diameter=0.0164, fin_thickness=0.001, fin_density=1/0.003,
                        pitch_normal=0.0313, pitch_parallel=0.0271,
    fin_height=0.0041,
   bundles_per_bay=1, parallel_bays=1, corbels=True)

    # All factors are matching again, except for the A min being different!
    # 5% diff, minor.

    h = h_ESDU_low_fin(m=0.914, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
         A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
         fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
         fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
         pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, 
         rho=1.217, Cp=1007., mu=1.8E-5, k=0.0253, k_fin=15)

    assert_allclose(h, 553.853836470948)
    
    h = h_ESDU_low_fin(m=0.914, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
         A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
         fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
         fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
         pitch_normal=AC.pitch_normal, pitch_parallel=AC.pitch_parallel, 
         rho=1.217, Cp=1007., mu=1.8E-5, k=0.0253, k_fin=15, Pr_wall=0.68)

    assert_allclose(h, 560.74807767957759)

def test_h_Briggs_Young():
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=20, tube_length=3, 
                        tube_diameter=1*inch, fin_thickness=0.000406, fin_density=1/0.002309,
                        pitch_normal=.06033, pitch_parallel=.05207,
                        fin_height=0.0159, tube_thickness=(.0254-.0186)/2,
                        bundles_per_bay=1, parallel_bays=1, corbels=True)

    h_bare_tube_basis = h_Briggs_Young(m=21.56, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
                             A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
                             fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
                             fin_thickness=AC.fin_thickness,
                             rho=1.161, Cp=1007., mu=1.85E-5, k=0.0263, k_fin=205)
    assert_allclose(h_bare_tube_basis, 1422.8722403237716)
    
    # Serth Process Heat Transfer Principles, Applications and Rules of Thumb 
    # example with a different correlation entirely
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=36*foot, 
                            tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                            angle=30, pitch_normal=2.5*inch, fin_height=0.625*inch, corbels=True)
    
    h = h_Briggs_Young(m=130.70315, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
                 A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
                 fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
                 fin_thickness=AC.fin_thickness,
                 rho=1.2013848, Cp=1009.0188, mu=1.9304793e-05, k=0.027864828, k_fin=238)
    
    # Goal. 51.785762
    # Back converting to their choice of basis - finned heat transfer coefficient
    # Very close answer
    assert_allclose(h/AC.A_increase/.853,  51.785762, atol=.3)
    
    
    
def test_h_Ganguli_VDI():
    
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=56, tube_length=36*foot, 
                            tube_diameter=1*inch, fin_thickness=0.013*inch, fin_density=10/inch,
                            angle=30, pitch_normal=2.5*inch, fin_height=0.625*inch, corbels=True)
    
    h = h_Ganguli_VDI(m=130.70315, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
                 A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
                 fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
                 fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
                pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal,
                 rho=1.2013848, Cp=1009.0188, mu=1.9304793e-05, k=0.027864828, k_fin=238)
    assert_allclose(h, 969.2850818578595)
    
    # Example in VDI
    # assumed in-line = angle 45, rest specified
    # VDI misses some parameters like fin tip area
    AC = AirCooledExchanger(tube_rows=4, tube_passes=1, tubes_per_row=17, tube_length=0.98, 
                            tube_diameter=1*inch, fin_thickness=0.4E-3, fin_density=9/inch,
                            angle=45, pitch_normal=0.06, fin_diameter=0.056)
    
    # Pr forced to match
    h = h_Ganguli_VDI(m=1.92, A=AC.A, A_min=AC.A_min, A_increase=AC.A_increase, A_fin=AC.A_fin,
                 A_tube_showing=AC.A_tube_showing, tube_diameter=AC.tube_diameter,
                 fin_diameter=AC.fin_diameter, bare_length=AC.bare_length,
                 fin_thickness=AC.fin_thickness, tube_rows=AC.tube_rows,
                pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal,
                 rho=0.909, Cp=1009.0188, mu=2.237e-05, k=0.03197131806799132, k_fin=209)
    # 22.49 goal, but there was a correction for velocity due to temperature increase
    # in the vdi answer
    assert_allclose(h/AC.A_increase, 22.49, rtol=2e-2)


def test_dP_ESDU_high_fin():
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=8, tube_length=0.5, 
        tube_diameter=0.0164, fin_thickness=0.001, fin_density=1/0.003,
        pitch_normal=0.0313, pitch_parallel=0.0271, fin_height=0.0041, corbels=True)

    dP = dP_ESDU_high_fin(m=0.914, A_min=AC.A_min, A_increase=AC.A_increase, flow_area_contraction_ratio=AC.flow_area_contraction_ratio, tube_diameter=AC.tube_diameter, pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal, tube_rows=AC.tube_rows, rho=1.217,  mu=0.000018)
    assert_allclose(dP, 485.6307687791502)


def test_dP_ESDU_low_fin():
    AC = AirCooledExchanger(tube_rows=4, tube_passes=4, tubes_per_row=8, tube_length=0.5, 
        tube_diameter=0.0164, fin_thickness=0.001, fin_density=1/0.003,
        pitch_normal=0.0313, pitch_parallel=0.0271, fin_height=0.0041, corbels=True)

    dP = dP_ESDU_low_fin(m=0.914, A_min=AC.A_min, A_increase=AC.A_increase, flow_area_contraction_ratio=AC.flow_area_contraction_ratio,
                     tube_diameter=AC.tube_diameter, fin_height=AC.fin_height, bare_length=AC.bare_length, pitch_parallel=AC.pitch_parallel, pitch_normal=AC.pitch_normal, tube_rows=AC.tube_rows, rho=1.217,  mu=0.000018)
    assert_allclose(dP, 464.54331418655914)
    # vs 414 Pa; Kf, ka almost perfect - A_min is the source of difference



def test_AirCooledExchangerPermutations():
    '''Demonstration of permutating all sorts of different options.
    
    need to try both nonlinear optimization (will work easier, need initial guesses) 
    and some form of discrete problem solution space.
    2 billion - that's just as many as there are floats! :)
    
    (len(tube_rows_options)*len(tube_passes_options)*len(tubes_per_row_options)
     *len(HTRI_Ls)*len(pitches)*len(fin_densities)
     *len(fin_heights)*len(ODs)*len(angles)
     *len(fin_thicknesses))/2.**31#15E-6/3600

    '''
    tube_rows_options = range(1, 20)
    tube_passes_options = range(1, 20)
    tubes_per_row_options = range(1, 40)
    
    from ht.hx import HTRI_Ls, HEDH_pitches
    from ht.air_cooler import fin_densities, fin_heights, ODs
    angles = [30, 45, 60, 90]
    pitches = sorted((set([j for k in HEDH_pitches.values() for j in k])))
    pitches = [1.25, 1.312, 1.33, 1.375, 1.4, 1.5]
    
    fin_thicknesses = [3E-4, 4E-4, 5E-4, 6E-4, 7E-4] # made up
    
    for i in range(100):
        angle = choice(angles)
        pitch = choice(pitches)
        tube_length = choice(HTRI_Ls)
        fin_density = choice(fin_densities)
        fin_height = choice(fin_heights)
        fin_thickness = choice(fin_thicknesses)
        tube_diameter = choice(ODs)
        
        tube_rows = choice(tube_rows_options)
        tube_passes = choice(tube_passes_options)
        tubes_per_row = choice(tubes_per_row_options)
        
        AC = AirCooledExchanger(tube_rows=tube_rows, tube_passes=tube_passes, 
                                tubes_per_row=tubes_per_row, tube_length=tube_length, 
                            tube_diameter=tube_diameter, fin_thickness=fin_thickness, fin_density=fin_density,
                            angle=angle, pitch=pitch, fin_height=fin_height)
    
        