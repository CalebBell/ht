'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, 2018 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

from math import pi

from fluids.core import Prandtl, Reynolds

from ht.conv_internal import laminar_entry_Seider_Tate

__all__ = ['Davis_David', 'Elamvaluthi_Srinivas', 'Groothuis_Hendal',
           'Hughmark', 'Knott', 'Kudirka_Grosh_McFadden', 'Martin_Sims',
           'Ravipudi_Godbold', 'Aggour',
           'h_two_phase', 'h_two_phase_methods']


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
        Quality at the specific tube interval [-]
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
    >>> Davis_David(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6,
    ... mul=1E-3)
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
    return Nu_TP*kl/D


def Elamvaluthi_Srinivas(m, x, D, rhol, rhog, Cpl, kl, mug, mu_b, mu_w=None):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and
    reviewed in [2]_.

    .. math::
        \frac{h_{TP} D}{k_L} = 0.5\left(\frac{\mu_G}{\mu_L}\right)^{0.25}
        Re_M^{0.7} Pr^{1/3}_L (\mu_b/\mu_w)^{0.14}

    .. math::
        Re_M = \frac{D V_L \rho_L}{\mu_L} + \frac{D V_g \rho_g}{\mu_g}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
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
        Viscosity of gas [Pa*s]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]

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
    >>> Elamvaluthi_Srinivas(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300,
    ... kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
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
    return Nu_TP*kl/D


def Groothuis_Hendal(m, x, D, rhol, rhog, Cpl, kl, mug, mu_b, mu_w=None,
                     water=False):
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
        Quality at the specific tube interval [-]
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
        Viscosity of gas [Pa*s]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]
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
    >>> Groothuis_Hendal(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6,
    ... mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
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
    return Nu_TP*kl/D


def Hughmark(m, x, alpha, D, L, Cpl, kl, mu_b=None, mu_w=None):
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
    alpha : float
        Void fraction in the tube, []
    D : float
        Diameter of the tube [m]
    L : float
        Length of the tube, [m]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]

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
    >>> Hughmark(m=1, x=.9, alpha=.9, D=.3, L=.5, Cpl=2300, kl=0.6, mu_b=1E-3,
    ... mu_w=1.2E-3)
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
    return Nu_TP*kl/D


def Knott(m, x, D, rhol, rhog, Cpl=None, kl=None, mu_b=None, mu_w=None, L=None,
          hl=None):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and
    reviewed in [2]_.

    Either a specified `hl` is required, or `Cpl`, `kl`, `mu_b`, `mu_w` and
    `L` are required to calculate `hl`.

    .. math::
        \frac{h_{TP}}{h_l} = \left(1 + \frac{V_{gs}}{V_{ls}}\right)^{1/3}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
    D : float
        Diameter of the tube [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    Cpl : float, optional
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float, optional
        Thermal conductivity of liquid [W/m/K]
    mu_b : float, optional
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]
    L : float, optional
        Length of the tube [m]
    hl : float, optional
        Liquid-phase heat transfer coefficient as described below, [W/m^2/K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    The liquid-only heat transfer coefficient will be calculated with the
    `laminar_entry_Seider_Tate` correlation, should it not be provided as an
    input. Many of the arguments to this function are optional and are only
    used if `hl` is not provided.

    `hl` should be calculated with a velocity equal to that determined with
    a combined volumetric flow of both the liquid and the gas. All other
    parameters used in calculating the heat transfer coefficient are those
    of the liquid. If the viscosity at the wall temperature is not given, the
    liquid viscosity correction in `laminar_entry_Seider_Tate` is not applied.

    Examples
    --------
    >>> Knott(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mu_b=1E-3,
    ... mu_w=1.2E-3, L=4)
    4225.536758045839

    References
    ----------
    .. [1] Knott, R. F., R. N. Anderson, Andreas. Acrivos, and E. E. Petersen.
       "An Experimental Study of Heat Transfer to Nitrogen-Oil Mixtures."
       Industrial & Engineering Chemistry 51, no. 11 (November 1, 1959):
       1369-72. doi:10.1021/ie50599a032.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L.
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with
       Seven Sets of Experimental Data, Including Flow Pattern and Tube
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1,
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vgs = m*x/(rhog*pi/4*D**2)
    Vls = m*(1-x)/(rhol*pi/4*D**2)
    if not hl:
        V = Vgs + Vls # Net velocity
        Re = Reynolds(V=V, D=D, rho=rhol, mu=mu_b)
        Pr = Prandtl(Cp=Cpl, k=kl, mu=mu_b)
        Nul = laminar_entry_Seider_Tate(Re=Re, Pr=Pr, L=L, Di=D, mu=mu_b, mu_w=mu_w)
        hl = Nul*kl/D
    return hl*(1 + Vgs/Vls)**(1/3.)


def Kudirka_Grosh_McFadden(m, x, D, rhol, rhog, Cpl, kl, mug, mu_b, mu_w=None):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and
    reviewed in [2]_.

    .. math::
        Nu = \frac{h_{TP} D}{k_l} = 125 \left(\frac{V_{gs}}{V_{ls}}
        \right)^{0.125}\left(\frac{\mu_g}{\mu_l}\right)^{0.6} Re_{ls}^{0.25}
        Pr_l^{1/3}\left(\frac{\mu_b}{\mu_w}\right)^{0.14}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
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
        Viscosity of gas [Pa*s]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    If the viscosity at the wall temperature is not given, the liquid viscosity
    correction is not applied.

    Developed for air-water and air-ethylene glycol systems with a L/D of 17.6
    and at low gas-liquid ratios. The flow regimes studied were bubble, slug,
    and froth flow.

    Examples
    --------
    >>> Kudirka_Grosh_McFadden(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300,
    ... kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    303.9941255903587

    References
    ----------
    .. [1] Kudirka, A. A., R. J. Grosh, and P. W. McFadden. "Heat Transfer in
       Two-Phase Flow of Gas-Liquid Mixtures." Industrial & Engineering
       Chemistry Fundamentals 4, no. 3 (August 1, 1965): 339-44.
       doi:10.1021/i160015a018.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L.
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with
       Seven Sets of Experimental Data, Including Flow Pattern and Tube
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1,
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vgs = m*x/(rhog*pi/4*D**2)
    Vls = m*(1-x)/(rhol*pi/4*D**2)
    Prl = Prandtl(Cp=Cpl, mu=mu_b, k=kl)
    Rels = D*Vls*rhol/mu_b
    Nu = 125*(Vgs/Vls)**0.125*(mug/mu_b)**0.6*Rels**0.25*Prl**(1/3.)
    if mu_w:
        Nu *= (mu_b/mu_w)**0.14
    return Nu*kl/D


def Martin_Sims(m, x, D, rhol, rhog, hl=None,
                Cpl=None, kl=None, mu_b=None, mu_w=None, L=None):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and
    reviewed in [2]_.

    .. math::
        \frac{h_{TP}}{h_l} = 1 + 0.64\sqrt{\frac{V_{gs}}{V_{ls}}}

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
    hl : float, optional
        Liquid-phase heat transfer coefficient as described below, [W/m^2/K]
    Cpl : float, optional
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float, optional
        Thermal conductivity of liquid [W/m/K]
    mu_b : float, optional
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]
    L : float, optional
        Length of the tube [m]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No specific suggestion for how to calculate the liquid-phase heat transfer
    coefficient is given in [1]_; [2]_ suggests to use the same procedure as
    in `Knott`.

    Examples
    --------
    >>> Martin_Sims(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, hl=141.2)
    5563.280000000001
    >>> Martin_Sims(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6,
    ... mu_b=1E-3, mu_w=1.2E-3, L=24)
    5977.505465781747

    References
    ----------
    .. [1] Martin, B. W, and G. E Sims. "Forced Convection Heat Transfer to
       Water with Air Injection in a Rectangular Duct." International Journal
       of Heat and Mass Transfer 14, no. 8 (August 1, 1971): 1115-34.
       doi:10.1016/0017-9310(71)90208-0.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L.
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with
       Seven Sets of Experimental Data, Including Flow Pattern and Tube
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1,
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vgs = m*x/(rhog*pi/4*D**2)
    Vls = m*(1-x)/(rhol*pi/4*D**2)
    if hl is None:
        V = Vgs + Vls # Net velocity
        Re = Reynolds(V=V, D=D, rho=rhol, mu=mu_b)
        Pr = Prandtl(Cp=Cpl, k=kl, mu=mu_b)
        Nul = laminar_entry_Seider_Tate(Re=Re, Pr=Pr, L=L, Di=D, mu=mu_b, mu_w=mu_w)
        hl = Nul*kl/D
    return hl*(1.0 + 0.64*(Vgs/Vls)**0.5)


def Ravipudi_Godbold(m, x, D, rhol, rhog, Cpl, kl, mug, mu_b, mu_w=None):
    r'''Calculates the two-phase non-boiling heat transfer coefficient of a
    liquid and gas flowing inside a tube of any inclination, as in [1]_ and
    reviewed in [2]_.

    .. math::
        Nu = \frac{h_{TP} D}{k_l} = 0.56 \left(\frac{V_{gs}}{V_{ls}}
        \right)^{0.3}\left(\frac{\mu_g}{\mu_l}\right)^{0.2} Re_{ls}^{0.6}
        Pr_l^{1/3}\left(\frac{\mu_b}{\mu_w}\right)^{0.14}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
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
        Viscosity of gas [Pa*s]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    If the viscosity at the wall temperature is not given, the liquid viscosity
    correction is not applied.

    Developed with a vertical pipe, superficial gas/liquid velocity ratios of
    1-90, in the froth regime, and for fluid mixtures of air and water,
    toluene, benzene, and methanol.

    Examples
    --------
    >>> Ravipudi_Godbold(m=1, x=.9, D=.3, rhol=1000, rhog=2.5, Cpl=2300, kl=.6, mug=1E-5, mu_b=1E-3, mu_w=1.2E-3)
    299.3796286459285

    References
    ----------
    .. [1] Ravipudi, S., and Godbold, T., The Effect of Mass Transfer on Heat
       Transfer Rates for Two-Phase Flow in a Vertical Pipe, Proceedings 6th
       International Heat Transfer Conference, Toronto, V. 1, p. 505-510, 1978.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L.
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with
       Seven Sets of Experimental Data, Including Flow Pattern and Tube
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1,
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vgs = m*x/(rhog*pi/4*D**2)
    Vls = m*(1-x)/(rhol*pi/4*D**2)
    Prl = Prandtl(Cp=Cpl, mu=mu_b, k=kl)
    Rels = D*Vls*rhol/mu_b
    Nu = 0.56*(Vgs/Vls)**0.3*(mug/mu_b)**0.2*Rels**0.6*Prl**(1/3.)
    if mu_w is not None:
        Nu *= (mu_b/mu_w)**0.14
    return Nu*kl/D


def Aggour(m, x, alpha, D, rhol, Cpl, kl, mu_b, mu_w=None, L=None,
           turbulent=None):
    r'''Calculates the two-phase non-boiling laminar heat transfer coefficient
    of a liquid and gas flowing inside a tube of any inclination, as in [1]_
    and reviewed in [2]_.

    Laminar for Rel <= 2000:

    .. math::
        h_{TP} = 1.615\frac{k_l}{D}\left(\frac{Re_l Pr_l D}{L}\right)^{1/3}
        \left(\frac{\mu_b}{\mu_w}\right)^{0.14}

    Turbulent for Rel > 2000:

    .. math::
        h_{TP} = 0.0155\frac{k_l}{D} Pr_l^{0.5} Re_l^{0.83}

    .. math::
        Re_l = \frac{\rho_l v_l D}{\mu_l}

    .. math::
        V_l = \frac{V_{ls}}{1-\alpha}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
    alpha : float
        Void fraction in the tube, [-]
    D : float
        Diameter of the tube [m]
    rhol : float
        Density of the liquid [kg/m^3]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mu_b : float
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]
    L : float, optional
        Length of the tube, [m]
    turbulent : bool or None, optional
        Whether or not to force the correlation to return the turbulent
        result; will return the laminar regime if False

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Developed with mixtures of air-water, helium-water, and freon-12-water and
    vertical tests. Studied flow patterns were bubbly, slug, annular,
    bubbly-slug, and slug-annular regimes. Superficial velocity ratios ranged
    from 0.02 to 470.

    A viscosity correction is only suggested for the laminar regime.
    If the viscosity at the wall temperature is not given, the liquid viscosity
    correction is not applied.

    Examples
    --------
    >>> Aggour(m=1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3)
    420.9347146885667

    References
    ----------
    .. [1] Aggour, Mohamed A. Hydrodynamics and Heat Transfer in Two-Phase
       Two-Component Flows, Ph.D. Thesis, University of Manutoba, Canada
       (1978). http://mspace.lib.umanitoba.ca/xmlui/handle/1993/14171.
    .. [2] Dongwoo Kim, Venkata K. Ryali, Afshin J. Ghajar, Ronald L.
       Dougherty. "Comparison of 20 Two-Phase Heat Transfer Correlations with
       Seven Sets of Experimental Data, Including Flow Pattern and Tube
       Inclination Effects." Heat Transfer Engineering 20, no. 1 (February 1,
       1999): 15-40. doi:10.1080/014576399271691.
    '''
    Vls = m*(1-x)/(rhol*pi/4*D**2)
    Vl = Vls/(1.-alpha)

    Prl = Prandtl(Cp=Cpl, k=kl, mu=mu_b)
    Rel = Reynolds(V=Vl, D=D, rho=rhol, mu=mu_b)

    if turbulent or (Rel > 2000.0 and turbulent is None):
        hl = 0.0155*(kl/D)*Rel**0.83*Prl**0.5
        return hl*(1-alpha)**-0.83
    else:
        hl = 1.615*(kl/D)*(Rel*Prl*D/L)**(1/3.)
        if mu_w:
            hl *= (mu_b/mu_w)**0.14
        return hl*(1.0 - alpha)**(-1/3.)


conv_two_phase_methods = {
    'Davis-David': (Davis_David, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mul')),
    'Elamvaluthi_Srinivas': (Elamvaluthi_Srinivas, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mug', 'mu_b', 'mu_w')),
    'Groothuis_Hendal': (Groothuis_Hendal, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mug', 'mu_b', 'mu_w')),
    'Hughmark': (Hughmark, ('m', 'x', 'alpha', 'D', 'L', 'Cpl', 'kl', 'mu_b', 'mu_w')),
    'Knott': (Knott, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mu_b', 'mu_w', 'L')),
    'Kudirka_Grosh_McFadden': (Kudirka_Grosh_McFadden, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mug', 'mu_b', 'mu_w')),
    'Martin_Sims': (Martin_Sims, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mu_b', 'mu_w', 'L')),
    'Ravipudi_Godbold': (Ravipudi_Godbold, ('m', 'x', 'D', 'rhol', 'rhog', 'Cpl', 'kl', 'mug', 'mu_b', 'mu_w')),
    'Aggour': (Aggour, ('m', 'x', 'alpha', 'D', 'rhol', 'Cpl', 'kl', 'mu_b', 'mu_w', 'L')),
}

# Not actually ranked
conv_two_phase_methods_ranked = ['Knott', 'Martin_Sims', 'Kudirka_Grosh_McFadden', 'Groothuis_Hendal',
                                 'Aggour', 'Hughmark', 'Elamvaluthi_Srinivas', 'Davis-David', 'Ravipudi_Godbold']

conv_two_phase_bad_method = "Correlation name not recognized; the availble methods are %s." %(list(conv_two_phase_methods.keys()))
"""Generating code:
for m in conv_two_phase_methods_ranked:
    (f_obj, args) = conv_two_phase_methods[m]
    print('elif method2 == "%s":' %(m))
    t = '    return %s(' %f_obj.__name__
    for ar in args:
        t += '%s=%s, ' %(ar, ar)
    t = t[:-2] + ')'
    print(t)

for m in conv_two_phase_methods_ranked:
    (f_obj, args) = conv_two_phase_methods[m]
    t = 'if '
    for ar in args:
        if ar not in ('m', 'x', 'D', 'Cpl', 'kl'):
            t += '%s is not None and ' %(ar)
    t = t[:-5] + ':'

    print(t)
    print('    methods.append("%s")' %m)
"""

def h_two_phase_methods(m, x, D, Cpl, kl, rhol=None, rhog=None, mul=None,
                        mu_b=None, mu_w=None, mug=None, L=None, alpha=None,
                        check_ranges=True):
    r'''Returns a list of correlation names for the case of two-phase
    non-boiling liquid-gas laminar flow heat transfer inside a tube.

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
    D : float
        Diameter of the tube [m]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    rhol : float, optional
        Density of the liquid [kg/m^3]
    rhog : float, optional
        Density of the gas [kg/m^3]
    mul : float, optional
        Viscosity of liquid [Pa*s]
    mu_b : float, optional
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]
    mug : float, optional
        Viscosity of gas [Pa*s]
    L : float, optional
        Length of the tube, [m]
    alpha : float, optional
        Void fraction in the tube, [-]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    A review of the correlations for which has the best performance has not
    been performed.

    Examples
    --------
    >>> h_two_phase_methods(m=1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=5)[0]
    'Aggour'
    '''
    methods = []
    if rhol is not None and rhog is not None and mu_b is not None and mu_w is not None and L is not None:
        methods.append("Knott")
    if rhol is not None and rhog is not None and mu_b is not None and mu_w is not None and L is not None:
        methods.append("Martin_Sims")
    if rhol is not None and rhog is not None and mug is not None and mu_b is not None and mu_w is not None:
        methods.append("Kudirka_Grosh_McFadden")
    if rhol is not None and rhog is not None and mug is not None and mu_b is not None and mu_w is not None:
        methods.append("Groothuis_Hendal")
    if alpha is not None and rhol is not None and mu_b is not None and mu_w is not None and L is not None:
        methods.append("Aggour")
    if alpha is not None and L is not None and mu_b is not None and mu_w is not None:
        methods.append("Hughmark")
    if rhol is not None and rhog is not None and mug is not None and mu_b is not None and mu_w is not None:
        methods.append("Elamvaluthi_Srinivas")
    if rhol is not None and rhog is not None and mul is not None:
        methods.append("Davis-David")
    if rhol is not None and rhog is not None and mug is not None and mu_b is not None and mu_w is not None:
        methods.append("Ravipudi_Godbold")
    return methods


def h_two_phase(m, x, D, Cpl, kl, rhol=None, rhog=None, mul=None,
                mu_b=None, mu_w=None, mug=None, L=None, alpha=None,
                method=None):
    r'''Calculates the two-phase non-boiling laminar heat transfer coefficient
    of a liquid and gas flowing inside a tube according to the specified
    method. Nine methods are available.


    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    x : float
        Quality at the specific tube interval [-]
    D : float
        Diameter of the tube [m]
    Cpl : float
        Constant-pressure heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    rhol : float, optional
        Density of the liquid [kg/m^3]
    rhog : float, optional
        Density of the gas [kg/m^3]
    mul : float, optional
        Viscosity of liquid [Pa*s]
    mu_b : float, optional
        Viscosity of liquid at bulk conditions (average of inlet/outlet
        temperature) [Pa*s]
    mu_w : float, optional
        Viscosity of liquid at wall temperature [Pa*s]
    mug : float, optional
        Viscosity of gas [Pa*s]
    L : float, optional
        Length of the tube, [m]
    alpha : float, optional
        Void fraction in the tube, [-]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Other Parameters
    ----------------
    method : string, optional
        A string of the function name to use, as in the dictionary
        conv_two_phase_methods.

    Notes
    -----
    It is difficult to compare correlations.

    Examples
    --------
    >>> h_two_phase(m=1, x=.9, D=.3, alpha=.9, rhol=1000, Cpl=2300, kl=.6, mu_b=1E-3, mu_w=1.2E-3, L=5, method='Aggour')
    420.9347146885667
    '''
    if method is None:
        method2 = h_two_phase_methods(m=m, x=x, D=D, Cpl=Cpl, kl=kl, rhol=rhol, rhog=rhog, mul=mul, mu_b=mu_b, mu_w=mu_w, mug=mug, L=L, alpha=alpha, check_ranges=True)[0]
    else:
        method2 = method

    if method2 == "Knott":
        return Knott(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mu_b=mu_b, mu_w=mu_w, L=L)
    elif method2 == "Martin_Sims":
        return Martin_Sims(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mu_b=mu_b, mu_w=mu_w, L=L)
    elif method2 == "Kudirka_Grosh_McFadden":
        return Kudirka_Grosh_McFadden(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mug=mug, mu_b=mu_b, mu_w=mu_w)
    elif method2 == "Groothuis_Hendal":
        return Groothuis_Hendal(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mug=mug, mu_b=mu_b, mu_w=mu_w)
    elif method2 == "Aggour":
        return Aggour(m=m, x=x, alpha=alpha, D=D, rhol=rhol, Cpl=Cpl, kl=kl, mu_b=mu_b, mu_w=mu_w, L=L)
    elif method2 == "Hughmark":
        return Hughmark(m=m, x=x, alpha=alpha, D=D, L=L, Cpl=Cpl, kl=kl, mu_b=mu_b, mu_w=mu_w)
    elif method2 == "Elamvaluthi_Srinivas":
        return Elamvaluthi_Srinivas(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mug=mug, mu_b=mu_b, mu_w=mu_w)
    elif method2 == "Davis-David":
        return Davis_David(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mul=mul)
    elif method2 == "Ravipudi_Godbold":
        return Ravipudi_Godbold(m=m, x=x, D=D, rhol=rhol, rhog=rhog, Cpl=Cpl, kl=kl, mug=mug, mu_b=mu_b, mu_w=mu_w)
    else:
        raise ValueError(conv_two_phase_bad_method)
