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
from scipy.constants import g, R
from math import sin, pi, log

__all__ = ['Boyko_Kruzhilin', 'Nusselt_laminar', 'h_kinetic', 
           'Akers_Deans_Crosser']

def Nusselt_laminar(Tsat, Tw, rhog, rhol, kl, mul, Hvap, L, angle=90):
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
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Optionally, the plate may be inclined.
    The constant 0.943 is actually:
    
    .. math::
        2\sqrt{2}/3

    Examples
    --------
    p. 578 in [1]_, matches exactly.

    >>> Nusselt_laminar(Tsat=370, Tw=350, rhog=7.0, rhol=585., kl=0.091,
    ... mul=158.9E-6, Hvap=776900, L=0.1)
    1482.206403453679

    References
    ----------
    .. [1] Hewitt, G. L. Shires T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    h = 2*2**0.5/3.*(kl**3*rhol*(rhol-rhog)*g*sin(angle/180.*pi)*Hvap/(mul*(Tsat-Tw)*L))**0.25
    return h


def Boyko_Kruzhilin(m, rhog, rhol, kl, mul, Cpl, D, x):
    r'''Calculates heat transfer coefficient for condensation
    of a pure chemical inside a vertical tube or tube bundle, as presented in
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
    h : float
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


def Akers_Deans_Crosser(m, rhog, rhol, kl, mul, Cpl, D, x):
    r'''Calculates heat transfer coefficient for condensation
    of a pure chemical inside a vertical tube or tube bundle, as presented in
    [2]_ according to [1]_.

    .. math::
        Nu = \frac{hD_i}{k_l} = C Re_e^n Pr_l^{1/3}
        
        C = 0.0265, n=0.8 \text{ for } Re_e > 5\times10^4
        
        C = 5.03, n=\frac{1}{3} \text{ for } Re_e < 5\times10^4

        Re_e = \frac{D_i G_e}{\mu_l}
        
        G_e = G\left[(1-x)+x(\rho_l/\rho_g)^{0.5}\right]
        
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
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----

    Examples
    --------
    >>> Akers_Deans_Crosser(m=0.35, rhog=6.36, rhol=582.9, kl=0.098, 
    ... mul=159E-6, Cpl=2520., D=0.03, x=0.85)
    7117.24177265201

    References
    ----------
    .. [1] Akers, W. W., H. A. Deans, and O. K. Crosser. "Condensing Heat 
       Transfer Within Horizontal Tubes." Chem. Eng. Progr. Vol: 55, Symposium 
       Ser. No. 29 (January 1, 1959).
    .. [2] Kakaç, Sadik, ed. Boilers, Evaporators, and Condensers. 1st. 
       Wiley-Interscience, 1991.
    '''
    G = m/(pi/4*D**2)
    Ge = G*((1-x) + x*(rhol/rhog)**0.5)
    Ree = D*Ge/mul
    Prl = mul*Cpl/kl
    if Ree > 5E4:
        C, n = 0.0265, 0.8
    else:
        C, n = 5.03, 1/3.
    Nu = C*Ree**n*Prl**(1/3.)
    h = Nu*kl/D
    return h

#print([Akers_Deans_Crosser(m=0.01, rhog=6.36, rhol=582.9, kl=0.098, mul=159E-6, Cpl=2520., D=0.03, x=0.85)])


def h_kinetic(T, P, MW, Hvap, f=1):
    r'''Calculates heat transfer coefficient for condensation
    of a pure chemical inside a vertical tube or tube bundle, as presented in
    [2]_ according to [1]_.

    .. math::
        h = \left(\frac{2f}{2-f}\right)\left(\frac{MW}{1000\cdot 2\pi R T}
        \right)^{0.5}\left(\frac{H_{vap}^2 P \cdot MW}{1000\cdot RT^2}\right)
        
    Parameters
    ----------
    T : float
        Vapor temperature, [K]
    P : float
        Vapor pressure, [Pa]
    MW : float
        Molecular weight of the gas, [g/mol]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    f : float
        Correction factor, [-]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    f is a correction factor for how the removal of gas particles affects the 
    behavior of the ideal gas in diffusing to the condensing surface. It is
    quite close to one, and has not been well explored in the literature due
    to the rarity of the importance of the kinetic resistance.

    Examples
    --------
    Water at 1 bar and 300 K:
    
    >>> h_kinetic(300, 1E5, 18.02, 2441674)
    30788845.562480535
    
    References
    ----------
    .. [1] Berman, L. D. "On the Effect of Molecular-Kinetic Resistance upon 
       Heat Transfer with Condensation." International Journal of Heat and Mass
       Transfer 10, no. 10 (October 1, 1967): 1463. 
       doi:10.1016/0017-9310(67)90033-6.
    .. [2] Kakaç, Sadik, ed. Boilers, Evaporators, and Condensers. 1 edition. 
       Wiley-Interscience, 1991.
    .. [3] Stephan, Karl. Heat Transfer in Condensation and Boiling. Translated
       by C. V. Green. Softcover reprint of the original 1st ed. 1992 edition. 
       Berlin; New York: Springer, 2013.
    '''
    h = (2*f)/(2-f)*(MW/(1000*2*pi*R*T))**0.5*(Hvap**2*P*MW)/(1000*R*T**2)
    return h
    
