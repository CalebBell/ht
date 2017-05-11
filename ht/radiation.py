# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
    return 2.*h*c**2/wavelength**5/(exp(h*c/(wavelength*T*k)) - 1.)


def q_rad(emissivity, T, T2=0):
    r'''Returns the radiant heat flux of a surface, optionally including
    assuming radiant heat transfer back to the surface.

    .. math::
        q = \epsilon \sigma (T_1^4 - T_2^4)

    Parameters
    ----------
    emissivity : float
        Fraction of black-body radiation which is emitted, []
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
    Emissivity must be less than 1. T2 may be larger than T.

    Examples
    --------
    >>> q_rad(emissivity=1, T=400)
    1451.613952

    >>> q_rad(.85, T=400, T2=305.)
    816.7821722650002

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    return sigma*emissivity*(T**4 - T2**4)
