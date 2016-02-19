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
from math import log
__all__ =['LMTD']

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
    ans : float
        log-mean temperature difference [K]

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
    ans = (dTF2-dTF1)/log(dTF2/dTF1)
    return ans
