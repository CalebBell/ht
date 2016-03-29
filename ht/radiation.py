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
from math import exp
from scipy.constants import sigma, h, c, k, pi
__all__ = ['blackbody_spectral_radiance', 'q_rad']

def blackbody_spectral_radiance(T, wavelength):
    r'''Returns the spectral radiance, in units of W/m^3/sr.

    .. math::
        I_{\lambda,blackbody,e}(\lambda,T)=\frac{2hc_o^2}
        {\lambda^5[\exp(hc_o/\lambda k T)-1]}

    Parameters
    ----------
    T : float
        Temperature of the surface, [K]
    wavelength : float
        Length of the wave to be considered, [m]

    Returns
    -------
    I : float
        Spectral radiance [W/m^3/sr]

    Notes
    -----
    Can be used to derive the Stefan-Boltzman law, or determine the maximum
    radiant frequency for a given temperature.

    Examples
    --------
    Checked with Spectral-calc.com, at [2]_.

    >>> blackbody_spectral_radiance(800., 4E-6)
    1311692056.2430143

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    .. [2] Spectral-calc.com. Blackbody Calculator, 2015.
       http://www.spectralcalc.com/blackbody_calculator/blackbody.php
    '''
    I = 2*h*c**2/wavelength**5/(exp(h*c/wavelength/T/k)-1)
    return I


def q_rad(emissivity, T, T2=0):
    r'''Returns the radiant heat flux of a surface, optionally including
    assuming radiant heat transfer back to the suface.

    .. math::
        q = \epsilon \sigma (T_1^4 - T_2^4)

    Parameters
    ----------
    emissivity : float
        Fraction of black-body radiation which is emmited, []
    T : float
        Temperature of the surface, [K]
    T2 : float, optional
        Temperature of the surrounding material of the surface [K]

    Returns
    -------
    q : float
        Heat exchange [W/m^2]

    Notes
    -----
    Emissivity must be less than 1. T2 mat be larger than T.

    Examples
    --------
    >>> q_rad(1., 400)
    1451.613952

    >>> q_rad(.85, 400, 305.)
    816.7821722650002

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    q = sigma*emissivity*(T**4-T2**4)
    return q
