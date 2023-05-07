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
SOFTWARE.
'''

from math import pi, sin

from fluids.constants import R, g
from fluids.core import Prandtl, Reynolds

from ht.conv_internal import turbulent_Dittus_Boelter

__all__ = ['Boyko_Kruzhilin', 'Nusselt_laminar', 'h_kinetic',
           'Akers_Deans_Crosser', 'Cavallini_Smith_Zecchin', 'Shah']


def Nusselt_laminar(Tsat, Tw, rhog, rhol, kl, mul, Hvap, L, angle=90.):
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
        Length of the plate [m]
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
    return 2.*2.**0.5/3.*(kl**3*rhol*(rhol - rhog)*g*sin(angle/180.*pi)
                          *Hvap/(mul*(Tsat - Tw)*L))**0.25


def Boyko_Kruzhilin(m, rhog, rhol, kl, mul, Cpl, D, x):
    r'''Calculates heat transfer coefficient for condensation
    of a pure chemical inside a vertical tube or tube bundle, as presented in
    [2]_ according to [1]_.

    .. math::
        h_f = h_{LO}\left[1 + x\left(\frac{\rho_L}{\rho_G} - 1\right)\right]^{0.5}

    .. math::
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
        Quality at the specific interval [-]

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
    return hlo*(1. + x*(rhol/rhog - 1.))**0.5


def Akers_Deans_Crosser(m, rhog, rhol, kl, mul, Cpl, D, x):
    r'''Calculates heat transfer coefficient for condensation
    of a pure chemical inside a vertical tube or tube bundle, as presented in
    [2]_ according to [1]_.

    .. math::
        Nu = \frac{hD_i}{k_l} = C Re_e^n Pr_l^{1/3}

    .. math::
        C = 0.0265, n=0.8 \text{ for } Re_e > 5\times10^4

    .. math::
        C = 5.03, n=\frac{1}{3} \text{ for } Re_e < 5\times10^4

    .. math::
        Re_e = \frac{D_i G_e}{\mu_l}

    .. math::
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
        Quality at the specific interval [-]

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
    return Nu*kl/D

#print([Akers_Deans_Crosser(m=0.01, rhog=6.36, rhol=582.9, kl=0.098, mul=159E-6, Cpl=2520., D=0.03, x=0.85)])


def h_kinetic(T, P, MW, Hvap, f=1.0):
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
    30788829.908851154

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
    return (2*f)/(2-f)*(MW/(1000*2*pi*R*T))**0.5*(Hvap**2*P*MW)/(1000*R*T**2)


def Cavallini_Smith_Zecchin(m, x, D, rhol, rhog, mul, mug, kl, Cpl):
    r'''Calculates heat transfer coefficient for condensation
    of a fluid inside a tube, as presented in
    [1]_, also given in [2]_ and [3]_.

    .. math::
        Nu = \frac{hD_i}{k_l} = 0.05 Re_e^{0.8} Pr_l^{0.33}

    .. math::
        Re_{eq} = Re_g(\mu_g/\mu_l)(\rho_l/\rho_g)^{0.5} + Re_l

    .. math::
        v_{gs} = \frac{mx}{\rho_g \frac{\pi}{4}D^2}

    .. math::
        v_{ls} = \frac{m(1-x)}{\rho_l \frac{\pi}{4}D^2}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific interval [-]
    D : float
        Diameter of the channel [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    mug : float
        Viscosity of gas [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----

    Examples
    --------
    >>> Cavallini_Smith_Zecchin(m=1, x=0.4, D=.3, rhol=800, rhog=2.5, mul=1E-5, mug=1E-3, kl=0.6, Cpl=2300)
    5578.218369177804

    References
    ----------
    .. [1] A. Cavallini, J. R. Smith and R. Zecchin, A dimensionless correlation
       for heat transfer in forced convection condensation, 6th International
       Heat Transfer Conference., Tokyo, Japan (1974) 309-313.
    .. [2] Kakaç, Sadik, ed. Boilers, Evaporators, and Condensers. 1st.
       Wiley-Interscience, 1991.
    .. [3] Balcilar, Muhammet, Ahmet Selim Dalkiliç, Berna Bolat, and Somchai
       Wongwises. "Investigation of Empirical Correlations on the Determination
       of Condensation Heat Transfer Characteristics during Downward Annular
       Flow of R134a inside a Vertical Smooth Tube Using Artificial
       Intelligence Algorithms." Journal of Mechanical Science and Technology
       25, no. 10 (October 12, 2011): 2683-2701. doi:10.1007/s12206-011-0618-2.
    '''
    Prl = Prandtl(Cp=Cpl, mu=mul, k=kl)
    Vl = m*(1-x)/(rhol*pi/4*D**2)
    Vg = m*x/(rhog*pi/4*D**2)
    Rel = Reynolds(V=Vl, D=D, rho=rhol, mu=mul)
    Reg = Reynolds(V=Vg, D=D, rho=rhog, mu=mug)
    """The following was coded, and may be used instead of the above lines,
    to check that the definitions of parameters here provide the same results
    as those defined in [1]_.
    G = m/(pi/4*D**2)
    Re = G*D/mul
    Rel = Re*(1-x)
    Reg = Re*x/(mug/mul)"""
    Reeq = Reg*(mug/mul)*(rhol/rhog)**0.5 + Rel
    Nul = 0.05*Reeq**0.8*Prl**0.33
    return Nul*kl/D # confirmed to be with respect to the liquid


def Shah(m, x, D, rhol, mul, kl, Cpl, P, Pc):
    r'''Calculates heat transfer coefficient for condensation
    of a fluid inside a tube, as presented in [1]_ and again by the same
    author in [2]_; also given in [3]_. Requires no properties of the gas.
    Uses the Dittus-Boelter correlation for single phase heat transfer
    coefficient, with a Reynolds number assuming all the flow is liquid.

    .. math::
        h_{TP} = h_L\left[(1-x)^{0.8} +\frac{3.8x^{0.76}(1-x)^{0.04}}
        {P_r^{0.38}}\right]

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific interval [-]
    D : float
        Diameter of the channel [m]
    rhol : float
        Density of the liquid [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    P : float
        Pressure of the fluid, [Pa]
    Pc : float
        Critical pressure of the fluid, [Pa]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ is well written an unambiguous as to how to apply this equation.

    Examples
    --------
    >>> Shah(m=1, x=0.4, D=.3, rhol=800, mul=1E-5, kl=0.6, Cpl=2300, P=1E6, Pc=2E7)
    2561.2593415479214

    References
    ----------
    .. [1] Shah, M. M. "A General Correlation for Heat Transfer during Film
       Condensation inside Pipes." International Journal of Heat and Mass
       Transfer 22, no. 4 (April 1, 1979): 547-56.
       doi:10.1016/0017-9310(79)90058-9.
    .. [2] Shah, M. M., Heat Transfer During Film Condensation in Tubes and
       Annuli: A Review of the Literature, ASHRAE Transactions, vol. 87, no.
       3, pp. 1086-1100, 1981.
    .. [3] Kakaç, Sadik, ed. Boilers, Evaporators, and Condensers. 1st.
       Wiley-Interscience, 1991.
    '''
    VL = m/(rhol*pi/4*D**2)
    ReL = Reynolds(V=VL, D=D, rho=rhol, mu=mul)
    Prl = Prandtl(Cp=Cpl, k=kl, mu=mul)
    hL = turbulent_Dittus_Boelter(ReL, Prl)*kl/D
    Pr = P/Pc
    return hL*((1-x)**0.8 + 3.8*x**0.76*(1-x)**0.04/Pr**0.38)

