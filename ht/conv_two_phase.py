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
from math import pi
from fluids import Reynolds, Prandtl

__all__ = ['Davis_David', 'Elamvaluthi_Srinivas', 'Groothuis_Hendal',
           'Hughmark']

def Davis_David(m, x, D, rhol, rhog, Cpl, kl, mul):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a 
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and 
    reviewed in [2]_.

    .. math::
        \frac{h_{TP} D}{k_l} = 0.060\left(\frac{\rho_L}{\rho_G}\right)^{0.28}
        \left(\frac{DG_{TP} x}{\mu_L}\right)^{0.87}
        \left(\frac{C_{p,L} \mu_L}{k_L}\right)^{0.4}
        
    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval []
    D : float
        Diameter of the tube [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mul : float
        Viscosity of liquid [Pa*s]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Developed for both vertical and horizontal flow, and flow patters of 
    annular or mist annular flow. Steam-water and air-water were the only 
    considered fluid combinations. Quality ranged from 0.1 to 1 in their data.
    [1]_ claimed an AAE of 17%.

    Examples
    --------
    >>> Davis_David(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mul=1E-3)
    1437.3282869955121

    References
    ----------
    .. [1] Davis, E. J., and M. M. David. "Two-Phase Gas-Liquid Convection Heat
       Transfer. A Correlation." Industrial & Engineering Chemistry 
       Fundamentals 3, no. 2 (May 1, 1964): 111-18. doi:10.1021/i160010a005.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L. 
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with 
       Seven Sets of Experimental Data, Including Flow Pattern and Tube 
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1, 
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    G = m/(pi/4*D**2)
    Prl = Prandtl(Cp=Cpl, mu=mul, k=kl)
    Nu_TP = 0.060*(rhol/rhog)**0.28*(D*G*x/mul)**0.87*Prl**0.4
    h = Nu_TP*kl/D
    return h


def Elamvaluthi_Srinivas(m, x, D, rhol, rhog, Cpl, kl, mug, mu_b, mu_w=None):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a 
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and 
    reviewed in [2]_.

    .. math::
        \frac{h_{TP} D}{k_L} = 0.5\left(\frac{\mu_G}{\mu_L}\right)^{0.25} 
        Re_M^{0.7} Pr^{1/3}_L (\mu_b/\mu_w)^{0.14}
        
        Re_M = \frac{D V_L \rho_L}{\mu_L} + \frac{D V_g \rho_g}{\mu_g}
        
    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval []
    D : float
        Diameter of the tube [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mug : float
        Viscosity of gas [Pa*S]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet 
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*S]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    If the viscosity at the wall temperature is not given, the liquid viscosity
    correction is not applied.
    
    Developed for vertical flow, and flow patters of bubbly and slug.
    Gas/liquid superficial velocity ratios from 0.3 to 4.6, liquid mass fluxes 
    from 200 to 1600 kg/m^2/s, and the fluids tested were air-water and 
    air-aqueous glycerine solutions. The tube inner diameter was 1 cm, and the
    L/D ratio was 86.
    
    Examples
    --------
    >>> Elamvaluthi_Srinivas(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    3901.2134471578584
    
    References
    ----------
    .. [1] Elamvaluthi, G., and N. S. Srinivas. "Two-Phase Heat Transfer in Two
       Component Vertical Flows." International Journal of Multiphase Flow 10, 
       no. 2 (April 1, 1984): 237-42. doi:10.1016/0301-9322(84)90021-1.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L. 
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with 
       Seven Sets of Experimental Data, Including Flow Pattern and Tube 
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1, 
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vg = m*x/(rhog*pi/4*D**2)
    Vl = m*(1-x)/(rhol*pi/4*D**2)

    Prl = Prandtl(Cp=Cpl, mu=mu_b, k=kl)
    ReM = D*Vl*rhol/mu_b + D*Vg*rhog/mug
    Nu_TP = 0.5*(mug/mu_b)**0.25*ReM**0.7*Prl**(1/3.)
    if mu_w:
        Nu_TP *= (mu_b/mu_w)**0.14
    h_TP = Nu_TP*kl/D
    return h_TP


def Groothuis_Hendal(m, x, D, rhol, rhog, Cpl, kl, mug, mu_b, mu_w=None, water=False):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a 
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and 
    reviewed in [2]_.

    .. math::
        Re_M = \frac{D V_{ls} \rho_l}{\mu_l} + \frac{D V_{gs} \rho_g}{\mu_g}
    
    For the air-water system:
    
    .. math::
        \frac{h_{TP} D}{k_L} = 0.029 Re_M^{0.87}Pr^{1/3}_l (\mu_b/\mu_w)^{0.14}
    
    For gas/air-oil systems (default):
    
    .. math::
        \frac{h_{TP} D}{k_L} = 2.6 Re_M^{0.39}Pr^{1/3}_l (\mu_b/\mu_w)^{0.14}
        
    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval []
    D : float
        Diameter of the tube [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mug : float
        Viscosity of gas [Pa*S]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet 
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*S]
    water : bool, optional
        Whether to use the water-air correlation or the gas/air-oil correlation

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    If the viscosity at the wall temperature is not given, the liquid viscosity
    correction is not applied.
    
    Developed for vertical pipes, with superficial velocity ratios of 0.6-250. 
    Tested fluids were air-water, and gas/air-oil. 
    
    Examples
    --------
    >>> Groothuis_Hendal(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    1192.9543445455754

    References
    ----------
    .. [1] Groothuis, H., and W. P. Hendal. "Heat Transfer in Two-Phase Flow.: 
       Chemical Engineering Science 11, no. 3 (November 1, 1959): 212-20. 
       doi:10.1016/0009-2509(59)80089-0. 
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L. 
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with 
       Seven Sets of Experimental Data, Including Flow Pattern and Tube 
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1, 
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vg = m*x/(rhog*pi/4*D**2)
    Vl = m*(1-x)/(rhol*pi/4*D**2)

    Prl = Prandtl(Cp=Cpl, mu=mu_b, k=kl)
    ReM = D*Vl*rhol/mu_b + D*Vg*rhog/mug

    if water:
        Nu_TP = 0.029*(ReM)**0.87*(Prl)**(1/3.)
    else:
        Nu_TP = 2.6*ReM**0.39*Prl**(1/3.)
    if mu_w:
        Nu_TP *= (mu_b/mu_w)**0.14
    h_TP = Nu_TP*kl/D
    return h_TP


def Hughmark(m, x, D, L, alpha, Cpl, kl, mu_b=None, mu_w=None):
    r'''Calculates the two-phase non-boiling laminar heat transfer coefficient  
    of a liquid and gas flowing inside a tube of any inclination, as in [1]_  
    and reviewed in [2]_.

    .. math::
        \frac{h_{TP} D}{k_l} = 1.75(1-\alpha)^{-0.5}\left(\frac{m_l C_{p,l}}
        {(1-\alpha)k_l L}\right)^{1/3}\left(\frac{\mu_b}{\mu_w}\right)^{0.14}
    
    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval []
    D : float
        Diameter of the tube [m]
    L : float
        Length of the tube, [m]
    alpha : float
        Void fraction in the tube, []
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet 
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*S]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    This model is based on a laminar entry length correlation - for a 
    sufficiently long tube, this will predict unrealistically low heat transfer
    coefficients.
    
    If the viscosity at the wall temperature is not given, the liquid viscosity
    correction is not applied.

    Developed for horizontal pipes in laminar slug flow. Data consisted of the 
    systems air-water, air-SAE 10 oil, gas-oil, air-diethylene glycol, and
    air-aqueous glycerine.
    
    Examples
    --------
    >>> Hughmark(m=1, x=.9, D=.3, L=.5, alpha=.9, Cpl=2300, kl=0.6, mu_b=1E-3, mu_w=1.2E-3)
    212.7411636127175

    References
    ----------
    .. [1] Hughmark, G. A. "Holdup and Heat Transfer in Horizontal Slug Gas-
       Liquid Flow." Chemical Engineering Science 20, no. 12 (December 1, 
       1965): 1007-10. doi:10.1016/0009-2509(65)80101-4.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L. 
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with 
       Seven Sets of Experimental Data, Including Flow Pattern and Tube 
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1, 
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    ml = m*(1-x)
    RL = 1-alpha
    Nu_TP = 1.75*(RL)**-0.5*(ml*Cpl/RL/kl/L)**(1/3.)
    if mu_b and mu_w:
        Nu_TP *= (mu_b/mu_w)**0.14
    h = Nu_TP*kl/D
    return h
