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


def Nu_plate_Kumar(Re, Pr, chevron_angle, mu=None, mu_wall=None):
    r'''Calculates Nusselt number for single-phase flow in a 
    **well-designed** Chevron-style plate heat exchanger according to [1]_.
    The data is believed to have been developed by APV International Limited,
    since acquired by SPX Corporation. This uses a curve fit of that data
    published in [2]_.
    
    .. math::
        Nu = C_1 Re^m Pr^{0.33}\left(\frac{\mu}{\mu_{wall}}\right)^{0.17}
        
    `C1` and `m` are coefficients looked up in a table, with varying ranges
    of Re validity and chevron angle validity. See the source for their
    exact values. The wall fluid property correction is included only if the
    viscosity values are provided.
        
    Parameters
    ----------
    Re : float
        Reynolds number with respect to the hydraulic diameter of the channels,
        [-]
    Pr : float
        Prandtl number calculated with bulk fluid properties, [-]
    chevron_angle : float
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. Many plate exchangers use two alternating patterns; use their
        average angle for that situation [degrees]
    mu : float, optional
        Viscosity of the fluid at the bulk (inlet and outlet average)
        temperature, [Pa*s]
    mu_wall : float, optional
        Viscosity of fluid at wall temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Dh`, [-]

    Notes
    -----
    Data on graph from Re=0.1 to Re=10000, with chevron angles 30 to 65 degrees.
    See `PlateExchanger` for further clarification on the definitions.
    
    It is believed the constants used in this correlation were curve-fit to
    the actual graph in [1]_ by the author of [2]_ as there is no 
    
    As the coefficients change, there are numerous small discontinuities, 
    although the data on the graphs is continuous with sharp transitions
    of the slope.
    
    The author of [1]_ states clearly this correlation is "applicable only to 
    well designed Chevron PHEs".
    
    Examples
    --------
    >>> Nu_plate_Kumar(Re=2000, Pr=0.7, chevron_angle=30)
    47.757818892853955
    
    With the wall-correction factor included:

    >>> Nu_plate_Kumar(Re=2000, Pr=0.7, chevron_angle=30, mu=1E-3, mu_wall=8E-4)
    49.604284135097544

    References
    ----------
    .. [1] Kumar, H. "The plate heat exchanger: construction and design." In 
       First U.K. National Conference on Heat Transfer: Held at the University 
       of Leeds, 3-5 July 1984, Institute of Chemical Engineering Symposium 
       Series, vol. 86, pp. 1275-1288. 1984.
    .. [2] Ayub, Zahid H. "Plate Heat Exchanger Literature Survey and New Heat
       Transfer and Pressure Drop Correlations for Refrigerant Evaporators." 
       Heat Transfer Engineering 24, no. 5 (September 1, 2003): 3-16. 
       doi:10.1080/01457630304056.
    '''
    # Uses the standard diameter as characteristic diameter
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
        
    Nu = C1*Re**m*Pr**0.33
    if mu_wall is not None and mu is not None:
        Nu *= (mu/mu_wall)**0.17
    return Nu
