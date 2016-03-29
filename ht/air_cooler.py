# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

from __future__ import division
from math import atan, sin
from ht.core import LMTD

__all__ = ['Ft_aircooler']

fin_densities_inch = [7, 8, 9, 10, 11] # fins/inch
fin_densities = [round(i/0.0254, 1) for i in fin_densities_inch]
ODs = [1, 1.25, 1.5, 2] # Actually though, just use TEMA. API 661 says 1 inch min.
fin_heights = [0.010, 0.012, 0.016] # m

tube_orientations = ['vertical  (bottom inlet)', 'vertical (top inlet)', 'horizontal', 'inclined']

_HTRI_fan_diameters = [0.71, 0.8, 0.9, 1.0, 1.2, 1.24, 1.385, 1.585, 1.78, 1.98, 2.22, 2.475, 2.775, 3.12, 3.515, 4.455, 4.95, 5.545, 6.24, 7.03, 7.92, 8.91, 9.9, 10.4, 11.1, 12.4, 13.85, 15.85]

fan_ring_types = ['straight', 'flanged',  'bell','15 degree cone', '30 degree cone']

fin_constructions = ['extruded', 'embedded', 'L-footed', 'overlapped L-footed', 'externally bonded', 'knurled footed']

headers = ['plug', 'removable cover', 'removable bonnet', 'welded bonnet']
configurations = ['forced draft', 'induced-draft (top drive)', 'induced-draft (bottom drive)']




# Coefs are from: Roetzel and Nicole - 1975 - Mean Temperature Difference for Heat Exchanger Design A General Approximate Explicit Equation
# Checked twice.

_crossflow_1_row_1_pass = [[-4.62E-1, -3.13E-2, -1.74E-1, -4.2E-2],
                           [5.08E0, 5.29E-1, 1.32E0, 3.47E-1],
                           [-1.57E1, -2.37E0, -2.93E0, -8.53E-1],
                           [1.72E1, 3.18E0, 1.99E0, 6.49E-1]]

_crossflow_2_rows_1_pass = [[-3.34E-1, -1.54E-1, -8.65E-2, 5.53E-2],
                            [3.3E0, 1.28E0, 5.46E-1, -4.05E-1],
                            [-8.7E0, -3.35E0, -9.29E-1, 9.53E-1],
                            [8.7E0, 2.83E0, 4.71E-1, -7.17E-1]]

_crossflow_3_rows_1_pass = [[-8.74E-2, -3.18E-2, -1.83E-2, 7.1E-3],
                            [1.05E0, 2.74E-1, 1.23E-1, -4.99E-2],
                            [-2.45E0, -7.46E-1, -1.56E-1, 1.09E-1],
                            [3.21E0, 6.68E-1, 6.17E-2, -7.46E-2]]

_crossflow_4_rows_1_pass = [[-4.14E-2, -1.39E-2, -7.23E-3, 6.1E-3],
                            [6.15E-1, 1.23E-1, 5.66E-2, -4.68E-2],
                            [-1.2E0, -3.45E-1, -4.37E-2, 1.07E-1],
                            [2.06E0, 3.18E-1, 1.11E-2, -7.57E-2]]

_crossflow_2_rows_2_pass = [[-2.35E-1, -7.73E-2, -5.98E-2, 5.25E-3],
                            [2.28E0, 6.32E-1, 3.64E-1, -1.27E-2],
                            [-6.44E0, -1.63E0, -6.13E-1, -1.14E-2],
                            [6.24E0, 1.35E0, 2.76E-1, 2.72E-2]]

_crossflow_3_rows_3_pass = [[-8.43E-1, 3.02E-2, 4.8E-1, 8.12E-2],
                            [5.85E0, -9.64E-3, -3.28E0, -8.34E-1],
                            [-1.28E1, -2.28E-1, 7.11E0, 2.19E0],
                            [9.24E0, 2.66E-1, -4.9E0, -1.69E0]]

_crossflow_4_rows_4_pass = [[-3.39E-1, 2.77E-2, 1.79E-1, -1.99E-2],
                            [2.38E0, -9.99E-2, -1.21E0, 4E-2],
                            [-5.26E0, 9.04E-2, 2.62E0, 4.94E-2],
                            [3.9E0, -8.45E-4, -1.81E0, -9.81E-2]]

_crossflow_4_rows_2_pass = [[-6.05E-1, 2.31E-2, 2.94E-1, 1.98E-2],
                            [4.34E0, 5.9E-3, -1.99E0, -3.05E-1],
                            [-9.72E0, -2.48E-1, 4.32, 8.97E-1],
                            [7.54E0, 2.87E-1, -3E0, -7.31E-1]]




def Ft_aircooler(Thi=None, Tho=None, Tci=None, Tco=None, Ntp=1, rows=1):
    r'''Calculates log-mean temperature difference correction factor for
    a crossflow heat exchanger, as in an Air Cooler. Method presented in [1]_,
    fit to other's nonexplicit work. Error is < 0.1%. Requires number of rows
    and tube passes as well as stream temperatures.

    .. math::
        F_T = 1 - \sum_{i=1}^m \sum_{k=1}^n a_{i,k}(1-r_{1,m})^k\sin(2i\arctan R)

        R = \frac{T_{hi} - T_{ho}}{T_{co}-T_{ci}}

        r_{1,m} = \frac{\Delta T_{lm}}{T_{hi} - T_{ci}}

    Parameters
    ----------
    Thi : float
        Temperature of hot fluid in [K]
    Tho : float
        Temperature of hot fluid out [K]
    Tci : float
        Temperature of cold fluid in [K]
    Tco : float
        Temperature of cold fluid out [K]

    Returns
    -------
    Ft : float
        Log-mean temperature difference correction factor []

    Notes
    -----
    This equation assumes that the hot fluid is tubeside, as in the case of air
    coolers. The model is not symmetric, so ensure to switch around the inputs
    if using this function for other purposes.

    This equation appears in [1]_. It has been verified.
    For some cases, approximations are made to match coefficients with the
    number of tube passes and rows provided.
    16 coefficients are used for each case; 8 cases are considered:

    * 1 row 1 pass
    * 2 rows 1 pass
    * 2 rows 2 passes
    * 3 rows 1 pass
    * 3 rows 3 passes
    * 4 rows 1 pass
    * 4 rows 2 passes
    * 4 rows 4 passes

    Examples
    --------
    Example 1 as in HTFS manual.
    Example 2 as in [1]_; author rounds to obtain a slightly different result.

    >>> Ft_aircooler(Thi=93., Tho=52., Tci=35, Tco=54.59, Ntp=2, rows=4)
    0.9570456123827129
    >>> Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)
    0.5505093604092708

    References
    ----------
    .. [1] Roetzel, W., and F. J. L. Nicole. "Mean Temperature Difference for
       Heat Exchanger Design-A General Approximate Explicit Equation." Journal
       of Heat Transfer 97, no. 1 (February 1, 1975): 5-8.
       doi:10.1115/1.3450288
    '''
    dTlm = LMTD(Thi=Thi, Tho=Tho, Tci=Tci, Tco=Tco)
    rlm = dTlm/(Thi-Tci)
    R = (Thi-Tho)/(Tco-Tci)
#    P = (Tco-Tci)/(Thi-Tci)

    if Ntp == 1 and rows == 1:
        coefs = _crossflow_1_row_1_pass
    elif Ntp == 1 and rows == 2:
        coefs = _crossflow_2_rows_1_pass
    elif Ntp == 1 and rows == 3:
        coefs = _crossflow_3_rows_1_pass
    elif Ntp == 1 and rows == 4:
        coefs = _crossflow_4_rows_1_pass
    elif Ntp == 1 and rows > 4:
        # A reasonable assumption
        coefs = _crossflow_4_rows_1_pass
    elif Ntp == 2 and rows == 2:
        coefs = _crossflow_2_rows_2_pass
    elif Ntp == 3 and rows == 3:
        coefs = _crossflow_3_rows_3_pass
    elif Ntp == 4 and rows == 4:
        coefs = _crossflow_4_rows_4_pass
    elif Ntp > 4 and rows > 4 and Ntp == rows:
        # A reasonable assumption
        coefs = _crossflow_4_rows_4_pass
    elif Ntp  == 2 and rows == 4:
        coefs = _crossflow_4_rows_2_pass
    else:
        # A bad assumption, but hey, gotta pick something.
        coefs = _crossflow_4_rows_2_pass
    tot = 0

    for k in range(len(coefs)):
        for i in range(len(coefs)):
            tot += coefs[k][i]*(1-rlm)**(k+1)*sin(2*(i+1)*atan(R))
    _Ft = 1-tot
    return _Ft


#print Ft_aircooler(Thi=66.82794, Tho=47.52647, Tci=15.4667, Tco=51.5889) # some example?
#print [Ft_aircooler(Thi=93., Tho=52., Tci=35, Tco=54.59, Ntp=2, rows=4)] # , 0.957045612383, works, _crossflow_4_rows_2_pass
#print [Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)] # , _crossflow_4_rows_1_pass, 0.550509360409, close enough
#print [[Ft_aircooler(Thi=125., Tho=80., Tci=25., Tco=95., Ntp=i, rows=j) for i in range(1,6)] for j in range(1, 6)] # , _crossflow_4_rows_1_pass, 0.550509360409, close enough