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
from math import log, pi, exp


def R_cylinder(Di, Do, k, L):
    r'''Returns the thermal resistance `R` of a cylinder of constant thermal
    conductivity `k`, of inner and outer diameter `Di` and `Do`, and with a
    length `L`.

    .. math::
        (hA)_{\text{cylinder}}=\frac{k}{\ln(D_o/D_i)} \cdot 2\pi L\\
        R_{\text{cylinder}}=\frac{1}{(hA)_{\text{cylinder}}}=
        \frac{\ln(D_o/D_i)}{2\pi Lk}

    Parameters
    ----------
    Di : float
        Inner diameter of the cylinder, [m]
    Do : float
        Outer diameter of the cylinder, [m]
    k : float
        Thermal conductivity of the cylinder, [W/m/K]
    L : float
        Length of the cylinder, [m]

    Returns
    -------
    R : float
        Thermal resistance [K/W]

    Examples
    --------
    >>> R_cylinder(0.9, 1., 20, 10)
    8.38432343682705e-05

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    hA = k*2*pi*L/log(Do/Di)
    R = 1./hA
    return R
