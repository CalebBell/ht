# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from ht.conv_plate import *
from numpy.testing import assert_allclose
import numpy as np
import pytest



 
def test_Nu_plate_Kumar():
    from fluids.friction import Kumar_beta_list
    from ht.conv_plate import Kumar_Nu_Res
    Nu = Nu_plate_Kumar(2000, 0.7, 30)
    assert_allclose(Nu, 47.757818892853955)
    
    Nu = Nu_plate_Kumar(Re=2000, Pr=0.7, chevron_angle=30, mu=1E-3, mu_wall=8E-4)
    assert_allclose(Nu, 49.604284135097544)
    
    all_ans_expected = [[[1.3741604132237337, 1.5167183720237427], [1.3741604132237337, 1.4917469901578877]],
     [[1.3741604132237337, 1.4917469901578877, 5.550501072445418, 5.686809480248301],
      [1.1640875871334992, 1.2445337163511674, 3.9101709259523125, 3.9566649343960067]],
     [[1.4929588988864342, 1.563892674590831, 7.514446806331191, 7.535921750318442],
      [1.3046449654318206, 1.3616258463940976, 5.549244219363172, 5.568849176342506]],
     [[1.3046449654318206, 1.3616258463940976, 6.464254426666383, 6.491074633865849],
      [1.3046449654318206, 1.360776035122095, 5.9841120030888915, 5.999181017513207]],
     [[1.3046449654318206, 1.360776035122095, 6.696608679807539, 6.712512276614001],
      [1.3046449654318206, 1.360776035122095, 6.696608679807539, 6.712512276614001]]]
     
    all_ans = []
    for i, beta_main in enumerate(Kumar_beta_list):
        beta_ans = []
        for beta in (beta_main-1, beta_main+1):
            Re_ans = []
            for Re_main in Kumar_Nu_Res[i]:
                for Re in [Re_main-1, Re_main+1]:
                    ans = Nu_plate_Kumar(Re, 0.7, beta)
                    Re_ans.append(ans)
            beta_ans.append(Re_ans)
        all_ans.append(beta_ans)
        
        for row1, row2 in zip(all_ans_expected, all_ans):
            assert_allclose(row1, row2)
    

def test_Nu_plate_Martin():
    Nu = Nu_plate_Martin(2000, .7, 1.18)
    assert_allclose(Nu, 43.5794551998615)
    
    Nu = Nu_plate_Martin(2000, .7, 1.18, variant='VDI')
    assert_allclose(Nu, 46.42246468447807)
    
    
def test_Nu_plate_Muley_Manglik():
    Nu = Nu_plate_Muley_Manglik(Re=2000, Pr=.7, chevron_angle=45, plate_enlargement_factor=1.18)
    assert_allclose(Nu, 36.49087100602062)
    
    
def test_Nu_plate_Khan_Khan():
    # The author presented three correlations; all of them are well matched by
    # the fourth correlation. beta max is not the largest angle in *your*
    # PHE, but of the ones they tested.
    Nu = Nu_plate_Khan_Khan(Re=1000, Pr=4.5, chevron_angle=30)
    assert_allclose(Nu,38.40883639103741 )


    
    
