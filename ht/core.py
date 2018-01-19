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
from math import log
__all__ =['LMTD', 'wall_factor']

def LMTD(Thi, Tho, Tci, Tco, counterflow=True):
    r'''Returns the log-mean temperature difference of an ideal counterflow
    or co-current heat exchanger.

    .. math::
        \Delta T_{LMTD}=\frac{\Delta T_1-\Delta T_2}{\ln(\Delta T_1/\Delta T_2)}

        \text{For countercurrent:      } \\
        \Delta T_1=T_{h,i}-T_{c,o}\\
        \Delta T_2=T_{h,o}-T_{c,i}

        \text{Parallel Flow Only:} \\
        {\Delta T_1=T_{h,i}-T_{c,i}}\\
        {\Delta T_2=T_{h,o}-T_{c,o}}

    Parameters
    ----------
    Thi : float
        Inlet temperature of hot fluid, [K]
    Tho : float
        Outlet temperature of hot fluid, [K]
    Tci : float
        Inlet temperature of cold fluid, [K]
    Tco : float
        Outlet temperature of cold fluid, [K]
    counterflow : bool, optional
        Whether the exchanger is counterflow or co-current

    Returns
    -------
    LMTD : float
        Log-mean temperature difference [K]

    Notes
    -----
    Any consistent set of units produces a consistent output.

    Examples
    --------
    Example 11.1 in [1]_.

    >>> LMTD(100., 60., 30., 40.2)
    43.200409294131525
    >>> LMTD(100., 60., 30., 40.2, counterflow=False)
    39.75251118049003

    References
    ----------
    .. [1] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ:
       Wiley, 2011.
    '''
    if counterflow:
        dTF1 = Thi-Tco
        dTF2 = Tho-Tci
    else:
        dTF1 = Thi-Tci
        dTF2 = Tho-Tco
    return (dTF2 - dTF1)/log(dTF2/dTF1)

WALL_FACTOR_VISCOSITY = 'Viscosity'
WALL_FACTOR_PRANDTL = 'Prandtl'
WALL_FACTOR_TEMPERATURE = 'Temperature'
WALL_FACTOR_DEFAULT = 'Default'

def _is_heating(prop, prop_wall):
    if prop_wall > prop:
        return False
    return True

def wall_factor(mu=None, mu_wall=None, Pr=None, Pr_wall=None, T=None, 
                T_wall=None, mu_heating_coeff=0.11, Pr_heating_coeff=0.11, 
                T_heating_coeff=0.11, mu_cooling_coeff=0.25, 
                Pr_cooling_coeff=0.25, T_cooling_coeff=0.25,
                property_option=WALL_FACTOR_PRANDTL):
    if property_option == WALL_FACTOR_DEFAULT:
        property_option = WALL_FACTOR_PRANDTL
    if property_option == WALL_FACTOR_VISCOSITY:
        if mu is None or mu_wall is None:
            raise Exception('Viscosity wall correction specified but both '
                            'viscosity values are not available.')
        heating = _is_heating(mu, mu_wall)
        if heating:
            return (mu/mu_wall)**mu_heating_coeff
        else:
            return (mu/mu_wall)**mu_cooling_coeff
    elif property_option == WALL_FACTOR_TEMPERATURE: 
        if T is None or T_wall is None:
            raise Exception('Temperature wall correction specified but both '
                            'temperature values are not available.')
        heating = _is_heating(T, T_wall)
        if heating:
            return (T/T_wall)**T_heating_coeff
        else:
            return (T/T_wall)**T_cooling_coeff
    elif property_option == WALL_FACTOR_PRANDTL: 
        if Pr is None or Pr_wall is None:
            raise Exception('Prandtl number wall correction specified but both'
                            ' Prandtl number values are not available.') 
        heating = _is_heating(Pr, Pr_wall)
        if heating:
            return (Pr/Pr_wall)**Pr_heating_coeff
        else:
            return (Pr/Pr_wall)**Pr_cooling_coeff
    else:
        raise Exception('Supported options are %s' %([WALL_FACTOR_VISCOSITY, 
                                                      WALL_FACTOR_PRANDTL, 
                                                      WALL_FACTOR_TEMPERATURE,
                                                      WALL_FACTOR_DEFAULT]))

