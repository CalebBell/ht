# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2018, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import pi
from fluids import Reynolds, Prandtl
from fluids.friction import Kumar_beta_list

__all__ = ['Nu_plate_Kumar']


Kumar_ms = [[0.349, 0.663],
      [0.349, 0.598, 0.663],
      [0.333, 0.591, 0.732],
      [0.326, 0.529, 0.703],
      [0.326, 0.503, 0.718]]

Kumar_C1s = [[0.718, 0.348],
       [0.718, 0.400, 0.300],
       [0.630, 0.291, 0.130],
       [0.562, 0.306, 0.108],
       [0.562, 0.331, 0.087]]

Kumar_Nu_Res = [[10.0],
          [10.0, 100.0],
          [20.0, 300.0],
          [20.0, 400.0],
          [20.0, 500.0]]


def Nu_plate_Kumar(Re, Pr, chevron_angle):
    # Uses the standard diameter as characteristic diameter
    # Applicable only to well designed Chevron PHEs
    # confirmed again to be the standard hydraulic diameter
    beta_list_len = len(Kumar_beta_list)
    
    for i in range(beta_list_len):
        if chevron_angle <= Kumar_beta_list[i]:
            C1_options, m_options, Re_ranges = Kumar_C1s[i], Kumar_ms[i], Kumar_Nu_Res[i]
            break
        elif i == beta_list_len-1:
            C1_options, m_options, Re_ranges = Kumar_C1s[-1], Kumar_ms[-1], Kumar_Nu_Res[-1]
                
    Re_len = len(Re_ranges)
    
    for j in range(Re_len):
        if Re <= Re_ranges[j]:
            C1, m = C1_options[j], m_options[j]
            break
        elif j == Re_len-1:
            C1, m = C1_options[-1], m_options[-1]
        
    # In fanning friction factor basis
    return C1*Re**m*Pr**0.33
