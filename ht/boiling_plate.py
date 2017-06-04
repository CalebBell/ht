# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from fluids import Reynolds, Prandtl, Bond


__all__ = ['h_boiling_Amalfi']

def h_boiling_Amalfi(m, x, Dh, rhol, rhog, mul, mug, kl, Hvap, sigma, q, 
                     A_channel_flow, chevron_angle=45):
    r'''Calculates the two-phase boiling heat transfer coefficient of a 
    liquid and gas flowing inside a plate and frame heat exchanger, as
    developed in [1]_ from a wide range of existing correlations and data sets. 
    Expected to be the most accurate correlation currently available.

    For Bond number < 4 (tiny channel case):
        
    .. math::
        h= 982 \left(\frac{k_l}{D_h}\right)\left(\frac{\beta}{\beta_{max}}\right)^{1.101}
        \left(\frac{G^2 D_h}{\rho_m \sigma}\right)^{0.315}
        \left(\frac{\rho_l}{\rho_g}\right)^{-0.224} Bo^{0.320}
        
    For Bond number >= 4:
        
    .. math::
        h = 18.495 \left(\frac{k_l}{D_h}\right) \left(\frac{\beta}{\beta_{max}}
        \right)^{0.248}\left(Re_g\right)^{0.135}\left(Re_{lo}\right)^{0.351}
        \left(\frac{\rho_l}{\rho_g}\right)^{-0.223} Bd^{0.235} Bo^{0.198}
        
    In the above equations, beta max is 45 degrees; Bo is Boiling number;
    and Bd is Bond number.
    
    Note that this model depends on the specific heat flux involved.
            
    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific point in the plate exchanger []
    Dh : float
        Hydraulic diameter of the plate, :math:`D_h = \frac{4\lambda}{\phi}` [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    mul : float
        Viscosity of the liquid [Pa*s]
    mug : float
        Viscosity of the gas [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Hvap : float
        Heat of vaporization of the fluid at the system pressure, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    q : float
        Heat flux, [W/m^2]
    A_channel_flow : float
        The flow area for the fluid, calculated as 
        :math:`A_{ch} = 2\cdot \text{width} \cdot \text{amplitude}` [m]
    chevron_angle : float, optional
        Angle of the plate corrugations with respect to the vertical axis
        (the direction of flow if the plates were straight), between 0 and
        90. For exchangers with two angles, use the average value. [degrees]

    Returns
    -------
    h : float
        Boiling heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Heat transfer correlation developed from 1903 datum. Fluids included R134a, 
    ammonia, R236fa, R600a, R290, R1270, R1234yf, R410A, R507A, ammonia/water, 
    and air/water mixtures. Wide range of operating conditions, plate geometries.
        
    Examples
    --------
    >>> h_boiling_Amalfi(m=3E-5, x=.4, Dh=0.00172, rhol=567., rhog=18.09, 
    ... kl=0.086, mul=156E-6, mug=7.11E-6, sigma=0.02, Hvap=9E5, q=1E5, 
    ... A_channel_flow=0.0003)
    776.0781179096225

    References
    ----------
    .. [1] Amalfi, Raffaele L., Farzad Vakili-Farahani, and John R. Thome. 
       "Flow Boiling and Frictional Pressure Gradients in Plate Heat Exchangers.
       Part 2: Comparison of Literature Methods to Database and New Prediction 
       Methods." International Journal of Refrigeration 61 (January 2016):
       185-203. doi:10.1016/j.ijrefrig.2015.07.009.
    '''    
    chevron_angle_max = 45.
    beta_s = chevron_angle/chevron_angle_max
    
    rho_s = (rhol/rhog) # rho start in model
    
    G = m/A_channel_flow # Calculating the area of the channel is normally specified well
    Bd = Bond(rhol=rhol, rhog=rhog, sigma=sigma, L=Dh)
    
    rho_h = 1./(x/rhog + (1-x)/rhol) # homogeneous holdup - mixture density calculation
    We_m = G*G*Dh/sigma/rho_h
    
    Bo = q/(G*Hvap) # Boiling number

    if Bd < 4:
        # Should occur normally for "microscale" conditions
        Nu_tp = 982*beta_s**1.101*We_m**0.315*Bo**0.320*rho_s**-0.224
    else:
        Re_lo = G*Dh/mul
        Re_g = G*x*Dh/mug
        Nu_tp = 18.495*beta_s**0.135*Re_g**0.135*Re_lo**0.351*Bd**0.235*Bo**0.198*rho_s**-0.223
    return kl/Dh*Nu_tp
