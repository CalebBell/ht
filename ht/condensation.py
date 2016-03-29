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
from scipy.constants import g
from math import sin, pi, log

__all__ = ['Boyko_Kruzhilin', 'Nusselt_laminar']

def Nusselt_laminar(Tsat=None, Tw=None, rhog=None, rhol=None, kl=None,
                    mul=None, Hvap=None, L=None, angle=90):
    r'''Calculates heat transfer coefficient for laminar film condensation
    of a pure chemical on a flat plate, as presented in [1]_ according to an
    analysis performed by Nusselt in 1916.

    .. math::
        h=0.943\left[\frac{g\sin(\theta)\rho_{liq}(\rho_l-\rho_v)k_{l}^3
        \Delta H_{vap}}{\mu_l(T_{sat}-T_w)L}\right]^{0.25}

    Parameters
    ----------
    Tsat : float
        Saturation temperature at operating pressure [Pa]
    Tw : float
        Wall temperature, [K]
    rhog : float
        Density of the gas [kg/m^3]
    rhol : float
        Density of the liquid [kg/m^3]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mul : float
        Viscosity of liquid [Pa*s]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    L : float
        Lenth of the plate [m]
    angle : float, optional
        Angle of inclination of the plate [degrees]

    Returns
    -------
    h: float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Optionally, the plate may be inclined.

    Examples
    --------
    p. 578 in [1]_, matches exactly.

    >>> Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091,
    ... mul=158.9E-6, Hvap=776900, L=0.1)
    1482.5066124858113

    References
    ----------
    .. [1] Hewitt, G. L. Shires T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    h = 0.943*(kl**3*rhol*(rhol-rhog)*g*sin(angle/180.*pi)*Hvap/(mul*(Tsat-Tw)*L))**0.25
    return h


def Boyko_Kruzhilin(m=None, rhog=None, rhol=None, kl=None, mul=None, Cpl=None,
                    D=None, x=None):
    r'''Calculates heat transfer coefficient for condensation
    of a pure chemical insude a vertical tube or tube bundle, as presented in
    [2]_ according to [1]_.

    .. math::
        h_f = h_{LO}\left[1 + x\left(\frac{\rho_L}{\rho_G} - 1\right)\right]^{0.5}

        h_{LO} = 0.021 \frac{k_L}{L} Re_{LO}^{0.8} Pr^{0.43}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    rhog : float
        Density of the gas [kg/m^3]
    rhol : float
        Density of the liquid [kg/m^3]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mul : float
        Viscosity of liquid [Pa*s]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    D : float
        Diameter of the tubing [m]
    x : float
        Quality at the specific interval []

    Returns
    -------
    h: float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    To calculate overall heat transfer coefficient during condensation,
    simply average values at x = 1 and x = 0.

    Examples
    --------
    Page 589 in [2]_, matches exactly.

    >>> Boyko_Kruzhilin(m=500*pi/4*.03**2, rhog=6.36, rhol=582.9, kl=0.098,
    ... mul=159E-6, Cpl=2520., D=0.03, x=0.85)
    10598.657227479956

    References
    ----------
    .. [1] Boyko, L. D., and G. N. Kruzhilin. "Heat Transfer and Hydraulic
       Resistance during Condensation of Steam in a Horizontal Tube and in a
       Bundle of Tubes." International Journal of Heat and Mass Transfer 10,
       no. 3 (March 1, 1967): 361-73. doi:10.1016/0017-9310(67)90152-4.
    .. [2] Hewitt, G. L. Shires T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    Vlo = m/rhol/(pi/4.*D**2)
    Relo = rhol*Vlo*D/mul
    Prl = mul*Cpl/kl
    hlo = 0.021*kl/D*Relo**0.8*Prl**0.43
    h = hlo*(1. + x*(rhol/rhog - 1.))**0.5
    return h

