# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
from math import log, log10
from fluids.constants import g

__all__ = ['Rohsenow', 'McNelly', 'Forster_Zuber', 'Montinsky',
'Stephan_Abdelsalam', 'HEDH_Taborek', 'Bier', 'Cooper', 'Gorenflo', 
'h_nucleic', 'Zuber', 'Serth_HEDH', 'HEDH_Montinsky', 'qmax_boiling', 
'h0_VDI_2e', 'h0_Gorenflow_1993', 'qmax_boiling_methods', 'h_nucleic_methods']


def Rohsenow(rhol, rhog, mul, kl, Cpl, Hvap, sigma, Te=None, q=None, Csf=0.013,
             n=1.7):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.
    
    Either heat flux or excess temperature is required.

    With `Te` specified:
    
    .. math::
        h = {{\mu }_{L}} \Delta H_{vap} \left[ \frac{g( \rho_L-\rho_v)}
        {\sigma } \right]^{0.5}\left[\frac{C_{p,L}\Delta T_e^{2/3}}{C_{sf}
        \Delta H_{vap} Pr_L^n}\right]^3
    
    With `q` specified:
    
    .. math::
        h = \left({{\mu }_{L}} \Delta H_{vap} \left[ \frac{g( \rho_L-\rho_v)}
        {\sigma } \right]^{0.5}\left[\frac{C_{p,L}\Delta T_e^{2/3}}{C_{sf}
        \Delta H_{vap} Pr_L^n}\right]^3\right)^{1/3}q^{2/3}

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    Csf : float
        Rohsenow coefficient specific to fluid and metal [-]
    n : float
        Constant, 1 for water, 1.7 (default) for other fluids usually [-]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No further work is required on this correlation. Multiple sources confirm
    its form and rearrangement.

    Examples
    --------
    h for water at atmospheric pressure on oxidized aluminum.

    >>> Rohsenow(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217,
    ... Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26)
    3723.655267067467

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Rohsenow, Warren M. "A Method of Correlating Heat Transfer Data for
       Surface Boiling of Liquids." Technical Report. Cambridge, Mass. : M.I.T.
       Division of Industrial Cooporation, 1951
    '''
    if Te:
        return mul*Hvap*(g*(rhol-rhog)/sigma)**0.5*(Cpl*Te**(2/3.)/Csf/Hvap/(Cpl*mul/kl)**n)**3
    elif q:
        A = mul*Hvap*(g*(rhol-rhog)/sigma)**0.5*(Cpl/Csf/Hvap/(Cpl*mul/kl)**n)**3
        return A**(1/3.)*q**(2/3.)
    else:
        raise Exception('Either q or Te is needed for this correlation')


def McNelly(rhol, rhog, kl, Cpl, Hvap, sigma, P, Te=None, q=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:
    
    .. math::
        h = \left(0.225\left(\frac{\Delta T_e C_{p,l}}{H_{vap}}\right)^{0.69}
        \left(\frac{P k_L}{\sigma}\right)^{0.31}
        \left(\frac{\rho_L}{\rho_V}-1\right)^{0.33}\right)^{1/0.31}

    With `q` specified:
    
    .. math::
        h = 0.225\left(\frac{q C_{p,l}}{H_{vap}}\right)^{0.69} \left(\frac{P 
        k_L}{\sigma}\right)^{0.31}\left(\frac{\rho_L}{\rho_V}-1\right)^{0.33}

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    P : float
        Saturation pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    
    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Further examples for this function are desired.

    Examples
    --------
    Water boiling, with excess temperature of 4.3 K.

    >>> McNelly(Te=4.3, P=101325, Cpl=4180., kl=0.688, sigma=0.0588,
    ... Hvap=2.25E6, rhol=958., rhog=0.597)
    533.8056972951352

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] McNelly M. J.: "A correlation of the rates of heat transfer to n
       ucleate boiling liquids," J. Imp Coll. Chem Eng Soc 7:18, 1953.
    '''
    if Te:
        return (0.225*(Te*Cpl/Hvap)**0.69*(P*kl/sigma)**0.31*(rhol/rhog-1.)**0.33
            )**(1./0.31)
    elif q:
        return 0.225*(q*Cpl/Hvap)**0.69*(P*kl/sigma)**0.31*(rhol/rhog-1.)**0.33
    else:
        raise Exception('Either q or Te is needed for this correlation')


def Forster_Zuber(rhol, rhog, mul, kl, Cpl, Hvap, sigma, dPsat, Te=None, q=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:
    
    .. math::
        h = 0.00122\left(\frac{k_L^{0.79} C_{p,l}^{0.45}\rho_L^{0.49}}
        {\sigma^{0.5}\mu_L^{0.29} H_{vap}^{0.24} \rho_V^{0.24}}\right)
        \Delta T_e^{0.24} \Delta P_{sat}^{0.75}

    With `q` specified:
    
    .. math::
        h = \left[0.00122\left(\frac{k_L^{0.79} C_{p,l}^{0.45}\rho_L^{0.49}}
        {\sigma^{0.5}\mu_L^{0.29} H_{vap}^{0.24} \rho_V^{0.24}}\right) \Delta 
        P_{sat}^{0.75} q^{0.24}\right]^{\frac{1}{1.24}}

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    dPsat : float
        Difference in saturation pressure of the fluid at Te and T, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Examples have been found in [1]_ and [3]_ and match exactly.

    Examples
    --------
    Water boiling, with excess temperature of 4.3K from [1]_.

    >>> Forster_Zuber(Te=4.3, dPsat=3906*4.3, Cpl=4180., kl=0.688,
    ... mul=0.275E-3, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)
    3519.9239897462644

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Forster, H. K., and N. Zuber. "Dynamics of Vapor Bubbles and Boiling
       Heat Transfer." AIChE Journal 1, no. 4 (December 1, 1955): 531-35.
       doi:10.1002/aic.690010425.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    if Te:
        return 0.00122*(kl**0.79*Cpl**0.45*rhol**0.49/sigma**0.5/mul**0.29/Hvap**0.24/rhog**0.24)*Te**0.24*dPsat**0.75
    elif q:
        return (0.00122*(kl**0.79*Cpl**0.45*rhol**0.49/sigma**0.5/mul**0.29/Hvap**0.24/rhog**0.24)*q**0.24*dPsat**0.75)**(1/1.24)
    else:
        raise Exception('Either q or Te is needed for this correlation')


def Montinsky(P, Pc, Te=None, q=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:
    
    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[1.8(P/P_c)^{0.17} + 
        4(P/P_c)^{1.2} + 10(P/P_c)^{10}\right]\right)^{1/0.3}

    With `q` specified:
    
    .. math::
        h = 0.00417P_c^{0.69} q^{0.7}\left[1.8(P/P_c)^{0.17} + 4(P/P_c)^{1.2} 
        + 10(P/P_c)^{10}\right]

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Formulas has been found consistent in all cited sources. Examples have
    been found in [1]_ and [3]_.

    The equation for this function is sometimes given with a constant of 3.7E-5
    instead of 0.00417 if critical pressure is not internally
    converted to kPa. [3]_ lists a constant of 3.596E-5.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3K from [1]_.

    >>> Montinsky(P=101325, Pc=22048321, Te=4.3)
    1185.0509770292663

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Mostinsky I. L.: "Application of the rule of corresponding states
       for the calculation of heat transfer and critical heat flux,"
       Teploenergetika 4:66, 1963 English Abstr. Br Chem Eng 8(8):586, 1963
    .. [3] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [4] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    if Te:
        return (0.00417*(Pc/1000.)**0.69*Te**0.7*(1.8*(P/Pc)**0.17 + 4*(P/Pc)**1.2
        +10*(P/Pc)**10))**(1/0.3)
    elif q:
        return (0.00417*(Pc/1000.)**0.69*q**0.7*(1.8*(P/Pc)**0.17 + 4*(P/Pc)**1.2
        +10*(P/Pc)**10))
    else:
        raise Exception('Either q or Te is needed for this correlation')


_angles_Stephan_Abdelsalam = {'general': 35, 'water': 45, 'hydrocarbon': 35,
'cryogenic': 1, 'refrigerant': 35}

def Stephan_Abdelsalam(rhol, rhog, mul, kl, Cpl, Hvap, sigma, Tsat, Te=None, 
                       q=None, kw=401, rhow=8.96, Cpw=384, angle=None, 
                       correlation='general'):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.
    Five variants are possible.

    Either heat flux or excess temperature is required. The forms for `Te` are
    not shown here, but are similar to those of the other functions.

    .. math::
        h = 0.23X_1^{0.674} X_2^{0.35} X_3^{0.371} X_5^{0.297} X_8^{-1.73} k_L/d_B

    .. math::
        X1 = \frac{q D_d}{K_L T_{sat}}

    .. math::
        X2 = \frac{\alpha^2 \rho_L}{\sigma D_d}

    .. math::
        X3 = \frac{C_{p,L} T_{sat} D_d^2}{\alpha^2}

    .. math::
        X4 = \frac{H_{vap} D_d^2}{\alpha^2}

    .. math::
        X5 = \frac{\rho_V}{\rho_L}

    .. math::
        X6 = \frac{C_{p,l} \mu_L}{k_L}

    .. math::
        X7 = \frac{\rho_W C_{p,W} k_W}{\rho_L C_{p,L} k_L}

    .. math::
        X8 = \frac{\rho_L-\rho_V}{\rho_L}

    .. math::
        D_b = 0.0146\theta\sqrt{\frac{2\sigma}{g(\rho_L-\rho_g)}}

    Respectively, the following four correlations are for water, hydrocarbons,
    cryogenic fluids, and refrigerants.

    .. math::
        h = 0.246\times 10^7 X1^{0.673} X4^{-1.58} X3^{1.26}X8^{5.22}k_L/d_B

    .. math::
        h = 0.0546 X5^{0.335} X1^{0.67} X8^{-4.33} X4^{0.248}k_L/d_B

    .. math::
        h = 4.82 X1^{0.624} X7^{0.117} X3^{0.374} X4^{-0.329}X5^{0.257} k_L/d_B

    .. math::
        h = 207 X1^{0.745} X5^{0.581} X6^{0.533} k_L/d_B

    Parameters
    ----------
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    Tsat : float
        Saturation temperature at operating pressure [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    kw : float, optional
        Thermal conductivity of wall (only for cryogenics) [W/m/K]
    rhow : float, optional
        Density of the wall (only for cryogenics) [kg/m^3]
    Cpw : float, optional
        Heat capacity of wall (only for cryogenics) [J/kg/K]
    angle : float, optional
        Contact angle of bubble with wall [degrees]
    correlation : str, optional
        Any of 'general', 'water', 'hydrocarbon', 'cryogenic', or 'refrigerant'

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    If cryogenic correlation is selected, metal properties are used. Default
    values are the properties of copper at STP.

    The angle is selected automatically if a correlation is selected; if angle
    is provided anyway, the automatic selection is ignored. A IndexError
    exception is raised if the correlation is not in the dictionary
    _angles_Stephan_Abdelsalam.

    Examples
    --------
    Example is from [3]_ and matches.

    >>> Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,
    ... sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, angle=35)
    26722.441071108373

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Stephan, K., and M. Abdelsalam. "Heat-Transfer Correlations for
       Natural Convection Boiling." International Journal of Heat and Mass
       Transfer 23, no. 1 (January 1980): 73-87.
       doi:10.1016/0017-9310(80)90140-4.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    if Te is None and q is None:
        raise Exception('Either q or Te is needed for this correlation')
    angle = _angles_Stephan_Abdelsalam[correlation]

    db = 0.0146*angle*(2*sigma/g/(rhol-rhog))**0.5
    diffusivity_L = kl/rhol/Cpl

    if Te:
        X1 = db/kl/Tsat*Te
    else:
        X1 = db/kl/Tsat*q
    X2 = diffusivity_L**2*rhol/sigma/db
    X3 = Hvap*db**2/diffusivity_L**2
    X4 = Hvap*db**2/diffusivity_L**2
    X5 = rhog/rhol
    X6 = Cpl*mul/kl
    X7 = rhow*Cpw*kw/(rhol*Cpl*kl)
    X8 = (rhol-rhog)/rhol

    if correlation == 'general':
        if Te:
            h = (0.23*X1**0.674*X2**0.35*X3**0.371*X5**0.297*X8**-1.73*kl/db)**(1/0.326)
        else:
            h = (0.23*X1**0.674*X2**0.35*X3**0.371*X5**0.297*X8**-1.73*kl/db)
    elif correlation == 'water':
        if Te:
            h = (0.246E7*X1**0.673*X4**-1.58*X3**1.26*X8**5.22*kl/db)**(1/0.327)
        else:
            h = (0.246E7*X1**0.673*X4**-1.58*X3**1.26*X8**5.22*kl/db)
    elif correlation == 'hydrocarbon':
        if Te:
            h = (0.0546*X5**0.335*X1**0.67*X8**-4.33*X4**0.248*kl/db)**(1/0.33)
        else:
            h = (0.0546*X5**0.335*X1**0.67*X8**-4.33*X4**0.248*kl/db)
    elif correlation == 'cryogenic':
        if Te:
            h = (4.82*X1**0.624*X7**0.117*X3**0.374*X4**-0.329*X5**0.257*kl/db)**(1/0.376)
        else:
            h = (4.82*X1**0.624*X7**0.117*X3**0.374*X4**-0.329*X5**0.257*kl/db)
    else:
        if Te:
            h = (207*X1**0.745*X5**0.581*X6**0.533*kl/db)**(1/0.255)
        else:
            h = (207*X1**0.745*X5**0.581*X6**0.533*kl/db)
    return h


def HEDH_Taborek(P, Pc, Te=None, q=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to Taborek (1986)
    as described in [1]_ and as presented in [2]_. Modification of [3]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:
    
    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[2.1P_r^{0.27} +
        \left(9 + (1-Pr^2)^{-1}\right)P_r^2 \right]\right)^{1/0.3}

    With `q` specified:
    
    .. math::
        h = 0.00417P_c^{0.69} q^{0.7}\left[2.1P_r^{0.27} + \left(9 + (1-Pr^2
        )^{-1}\right)P_r^2\right]

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Example is from [3]_ and matches to within the error of the algebraic
    manipulation rounding.

    Examples
    --------
    >>> HEDH_Taborek(Te=16.2, P=310.3E3, Pc=2550E3)
    1397.272486525486

    References
    ----------
    .. [1] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [2] Mostinsky I. L.: "Application of the rule of corresponding states
       for the calculation of heat transfer and critical heat flux,"
       Teploenergetika 4:66, 1963 English Abstr. Br Chem Eng 8(8):586, 1963
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    Pr = P/Pc
    if Te:
        return (0.00417*(Pc/1000.)**0.69*Te**0.7*(2.1*Pr**0.27
        + (9 + 1./(1-Pr**2))*Pr**2))**(1/0.3)
    elif q:
        return (0.00417*(Pc/1000.)**0.69*q**0.7*(2.1*Pr**0.27
        + (9 + 1./(1-Pr**2))*Pr**2))
    else:
        raise Exception('Either q or Te is needed for this correlation')


def Bier(P, Pc, Te=None, q=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [1]_ .

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[0.7 + 2P_r\left(4 +
        \frac{1}{1-P_r}\right)  \right]\right)^{1/0.3}

    With `q` specified:

    .. math::
        h = 0.00417P_c^{0.69} \Delta q^{0.7}\left[0.7 + 2P_r\left(4 + 
        \frac{1}{1-P_r}\right)  \right]

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No examples of this are known. Seems to give very different results than
    other correlations.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3 K from [1]_.

    >>> Bier(101325., 22048321.0, Te=4.3)
    1290.5349471503353

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    Pr = P/Pc
    if Te:
        return (0.00417*(Pc/1000.)**0.69*Te**0.7*(0.7 + 2.*Pr*(4. + 1./(1.-Pr))))**(1./0.3)
    elif q:
        return 0.00417*(Pc/1000.)**0.69*q**0.7*(0.7 + 2.*Pr*(4. + 1./(1. - Pr)))
    else:
        raise Exception('Either q or Te is needed for this correlation')


def Cooper(P, Pc, MW, Te=None, q=None, Rp=1E-6):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    Either heat flux or excess temperature is required.

    With `Te` specified:

    .. math::
        h = \left(55\Delta Te^{0.67} \frac{P}{P_c}^{(0.12 - 0.2\log_{10} R_p)}
        (-\log_{10} \frac{P}{P_c})^{-0.55} MW^{-0.5}\right)^{1/0.33}

    With `q` specified:

    .. math::
        h = 55q^{0.67} \frac{P}{P_c}^{(0.12 - 0.2\log_{10} R_p)}(-\log_{10} 
        \frac{P}{P_c})^{-0.55} MW^{-0.5}

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    MW : float
        Molecular weight of fluid, [g/mol]
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    Rp : float
        Roughness parameter of the surface (1 micrometer default), [m]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Examples 1 and 2 are for water and benzene, from [1]_.
    Roughness parameter is with an old definition. Accordingly, it is
    not used by the h function.
    If unchanged, the roughness parameter's logarithm gives a value of 0.12
    as an exponent of reduced pressure.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3 K from [1]_.

    >>> Cooper(P=101325., Pc=22048321.0, MW=18.02, Te=4.3)
    1558.1435442153575

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] M. G. Cooper, "Saturation and Nucleate Pool Boiling: A Simple
       Correlation," Inst. Chem. Eng. Syrup. Ser. (86/2): 785, 1984.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    Rp*= 1E6
    if Te:
        return (55*Te**0.67*(P/Pc)**(0.12 - 0.2*log10(Rp))*(
             -log10(P/Pc))**-0.55*MW**-0.5)**(1/0.33)
    elif q:
        return (55*q**0.67*(P/Pc)**(0.12 - 0.2*log10(Rp))*(
             -log10(P/Pc))**-0.55*MW**-0.5)
    else:
        raise Exception('Either q or Te is needed for this correlation')


h0_Gorenflow_1993 = {'74-82-8': 7000.0, '74-84-0': 4500.0, '74-98-6': 4000.0, 
'106-97-8': 3600.0, '109-66-0': 3400.0, '78-78-4': 2500.0, '110-54-3': 3300.0, 
'142-82-5': 3200.0, '71-43-2': 2900.0, '108-88-3': 2800.0, '92-52-4': 2100.0, 
'67-56-1': 5400.0, '64-17-5': 4400.0, '71-23-8': 3800.0, '67-63-0': 3000.0, 
'71-36-3': 2600.0, '78-83-1': 4500.0, '67-64-1': 3300.0, '75-69-4': 2800.0, 
'75-71-8': 4000.0, '75-72-9': 3900.0, '75-63-8': 3500.0, '75-45-6': 3900.0, 
'75-46-7': 4400.0, '76-13-1': 2650.0, '76-14-2': 3800.0, '76-15-3': 3200.0, 
'811-97-2': 4500.0, '28987-04-4': 3700.0, '431-89-0': 3800.0, '115-25-3': 4200.0, 
'74-87-3': 4400.0, '56-23-5': 3200.0, '75-73-0': 4750.0, '7732-18-5': 5600.0, 
'7664-41-7': 7000.0, '124-38-9': 5100.0, '2551-62-4': 3700.0, '7782-44-7': 9500.0, 
'7727-37-9': 10000.0, '7440-37-1': 8200.0, '7440-01-9': 20000.0, '1333-74-0': 24000.0, 
'7440-59-7': 2000.0}


def Gorenflo(P, Pc, q=None, Te=None, CASRN=None, h0=None, Ra=4E-7):
    r'''Calculates heat transfer coefficient for a pool boiling according to 
    [1]_ and also presented in [2]_. Calculation is based on the corresponding
    states law, with a single regression constant per fluid. P and Pc are 
    always required.
    
    Either `q` or `Te` may be specified. Either `CASRN` or `h0` may be 
    specified as well. If `CASRN` is specified and the fluid is not in the
    list of those studied, an error is raises.

    .. math::
        \frac{h}{h_0} = C_W F(p^*) \left(\frac{q}{q_0}\right)^n
        
    .. math::
        C_W = \left(\frac{R_a}{R_{ao}}\right)^{0.133}

    .. math::
        q_0 = 20 \;000 \frac{\text{W}}{\text{m}^{2}}
        
    .. math::
        R_{ao} = 0.4 \mu\text{m}
        
    For fluids other than water:
    
    .. math::
        n = 0.9 - 0.3 p^{*0.3}
        
    .. math::
        f(p^*) = 1.2p^{*0.27} + \left(2.5 + \frac{1}{1-p^*}\right)p^*
    
    For water:
    
    .. math::
        n = 0.9 - 0.3 p^{*0.15}
        
    .. math::
        f(p^*) = 1.73p^{*0.27} + \left(6.1 + \frac{0.68}{1-p^*}\right)p^2

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    q : float, optional
        Heat flux, [W/m^2]
    Te : float, optional
        Excess wall temperature, [K]
    CASRN : str, optional
        CASRN of fluid
    h0 : float
        Reference heat transfer coefficient, [W/m^2/K]
    Ra : float, optional
        Roughness parameter of the surface (0.4 micrometer default), [m]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    A more recent set of reference heat fluxes is available. Where a range of
    values was listed for reference heat fluxes in [1]_, values from the
    second edition of [1]_ were used instead. 44 values are available, all
    listed in the dictionary `h0_Gorenflow_1993`. Values range from 2000
    to 24000 W/m^2/K.

    Examples
    --------
    Water boiling at 3 bar and a heat flux of 2E4 W/m^2/K.

    >>> Gorenflo(3E5, 22048320., q=2E4, CASRN='7732-18-5')
    3043.344595525422

    References
    ----------
    .. [1] Schlunder, Ernst U, VDI. VDI Heat Atlas. Dusseldorf: V.D.I. Verlag, 
       1993. http://digital.ub.uni-paderborn.de/hs/download/pdf/41898?originalFilename=true
    .. [2] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella. 
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    Pr = P/Pc
    Ra0 = 0.4E-6
    q0 = 2E4
    if CASRN not in h0_Gorenflow_1993 and h0 is None:
        raise Exception('Reference heat transfer coefficient not known')
    if not h0:
        h0 = h0_Gorenflow_1993[CASRN]
    if CASRN != '7732-18-5':
        # Case for not dealing with water
        n = 0.9 - 0.3*Pr**0.3
        Fp = 1.2*Pr**0.27 + (2.5 + 1/(1-Pr))*Pr
    else:
        # Case for water
        n = 0.9 - 0.3*Pr**0.15
        Fp = 1.73*Pr**0.27 + (6.1 + 0.68/(1-Pr))*Pr**2
    CW = (Ra/Ra0)**0.133
    if q:
        return h0*CW*Fp*(q/q0)**n
    elif Te:
        A = h0*CW*Fp*(Te/q0)**n
        return A**(-1./(n - 1.))
    else:
        raise Exception('Either q or Te is needed for this correlation')
        

h0_VDI_2e = {'74-82-8': 7200.0, '74-85-1': 4200.0, '74-84-0': 4600.0, 
'115-07-1': 4200.0, '74-98-6': 4300.0, '106-97-8': 3600.0, '75-28-5': 3700.0, 
'109-66-0': 3300.0, '78-78-4': 3200.0, '110-54-3': 3200.0, '110-82-7': 3000.0, 
'142-82-5': 2900.0, '71-43-2': 2900.0, '108-88-3': 2800.0, '92-52-4': 2100.0, 
'67-56-1': 5400.0, '64-17-5': 4350.0, '71-23-8': 3750.0, '67-63-0': 4100.0, 
'71-36-3': 2600.0, '78-83-1': 4500.0, '78-92-2': 3400.0, '75-07-0': 3500.0, 
'67-64-1': 3300.0, '124-38-9': 5500.0, '75-46-7': 4800.0, '75-10-5': 5000.0, 
'354-33-6': 4400.0, '811-97-2': 4200.0, '420-46-2': 4700.0, '75-37-6': 4600.0, 
'754-12-1': 3000.0, '431-89-0': 4100.0, '115-25-3': 4200.0, '75-73-0': 4750.0, 
'306-83-2': 3000.0, '75-69-4': 2800.0, '75-71-8': 4000.0, '75-72-9': 3900.0, 
'75-63-8': 3500.0, '75-45-6': 3900.0, '76-13-1': 2650.0, '76-14-2': 3800.0, 
'76-15-3': 4200.0, '74-87-3': 4400.0, '56-23-5': 3200.0, '2551-62-4': 3700.0, 
'7732-18-5': 5600.0, '7664-41-7': 7000.0, '7782-44-7': 9500.0, '7727-37-9': 10000.0, 
'7440-37-1': 8200.0, '7440-01-9': 20000.0, '1333-74-0': 24000.0, '7440-59-7': 2000.0}



cryogenics = {'132259-10-0': 'Air', '7440-37-1': 'Argon', '630-08-0':
'carbon monoxide', '7782-39-0': 'deuterium', '7782-41-4': 'fluorine',
'7440-59-7': 'helium', '1333-74-0': 'hydrogen', '7439-90-9': 'krypton',
'74-82-8': 'methane', '7440-01-9': 'neon', '7727-37-9': 'nitrogen',
'7782-44-7': 'oxygen', '7440-63-3': 'xenon'}

h_nucleic_methods = ['Stephan-Abdelsalam', 'Stephan-Abdelsalam water', 
                     'Stephan-Abdelsalam cryogenic', 'HEDH-Taborek', 
                     'Forster-Zuber', 'Rohsenow', 'Cooper', 'Bier',
                     'Montinsky', 'McNelly', 'Gorenflo (1993)']


def h_nucleic(Te=None, q=None, Tsat=None, P=None, dPsat=None, Cpl=None, 
              kl=None, mul=None, rhol=None, sigma=None, Hvap=None, rhog=None, 
              MW=None, Pc=None, CAS=None, Method=None, AvailableMethods=False, 
              **kwargs):
    r'''This function handles the calculation of nucleate boiling
    heat flux and chooses the best method for performing the calculation
    based on the provided information.

    One of `Te` and `q` are always required.

    Parameters
    ----------
    Te : float, optional
        Excess wall temperature, [K]
    q : float, optional
        Heat flux, [W/m^2]
    Tsat : float, optional
        Saturation temperature at operating pressure [Pa]
    P : float, optional
        Saturation pressure of fluid, [Pa]
    dPsat : float, optional
        Difference in saturation pressure of the fluid at Te and T, [Pa]
    Cpl : float, optional
        Heat capacity of liquid [J/kg/K]
    kl : float, optional
        Thermal conductivity of liquid [W/m/K]
    mul : float, optional
        Viscosity of liquid [Pa*s]
    rhol : float, optional
        Density of the liquid [kg/m^3]
    sigma : float, optional
        Surface tension of liquid [N/m]
    Hvap : float, optional
        Heat of vaporization of the fluid at P, [J/kg]
    rhog : float, optional
        Density of the produced gas [kg/m^3]
    MW : float, optional
        Molecular weight of fluid, [g/mol]
    Pc : float, optional
        Critical pressure of fluid, [Pa]
    CAS : str, optional
        CAS of fluid

    Returns
    -------
    h : float
        Nucleate boiling heat flux [W/m^2]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate `h` with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The name of the method to use; one of ['Gorenflo (1993)', 
        'Stephan-Abdelsalam water', 'Stephan-Abdelsalam cryogenic', 
        'Stephan-Abdelsalam', 'HEDH-Taborek', 'Forster-Zuber', 'Rohsenow', 
        'Cooper', 'Bier', 'Montinsky', 'McNelly']
    AvailableMethods : bool, optional
        If True, function will consider which methods which can be used to
        calculate `h` with the given inputs
        
    Notes
    -----
    The methods Stephan-Abdelsalam, Cooper, and Gorenflo all take other 
    arguments as well such as surface roughness or the thermal properties of
    the wall material. See them for their documentation. These parameters
    can also be passed as keyword arguments.
    
    >>> h_nucleic(P=3E5, Pc=22048320., q=2E4, CAS='7732-18-5', Ra=1E-6)
    3437.7726419934147
    
    Examples
    --------
    Water boiling at 3 bar and a heat flux of 2E4 W/m^2/K.

    >>> h_nucleic(P=3E5, Pc=22048320., q=2E4, CAS='7732-18-5')
    3043.344595525422
    
    Water, known excess temperature of 4.9 K, Rohsenow method

    >>> h_nucleic(rhol=957.854, rhog=0.595593, mul=2.79E-4, kl=0.680, Cpl=4217,
    ... Hvap=2.257E6, sigma=0.0589, Te=4.9, Csf=0.011, n=1.26, 
    ... Method='Rohsenow')
    3723.655267067467
    '''
    def list_methods():
        methods = []
        if all((P, Pc)):
            if CAS and CAS in h0_Gorenflow_1993:
                methods.append('Gorenflo (1993)')
        if all((Te, Tsat, Cpl, kl, mul, sigma, Hvap, rhol, rhog)):
            if CAS and CAS == '7732-18-5':
                methods.append('Stephan-Abdelsalam water')
            if CAS and CAS in cryogenics:
                methods.append('Stephan-Abdelsalam cryogenic')
            methods.append('Stephan-Abdelsalam')
        if all((Te, P, Pc)):
            methods.append('HEDH-Taborek')
        if all((Te, dPsat, Cpl, kl, mul, sigma, Hvap, rhol, rhog)):
            methods.append('Forster-Zuber')
        if all((Te, Cpl, kl, mul, sigma, Hvap, rhol, rhog)):
            methods.append('Rohsenow')
        if all((Te, P, Pc, MW)):
            methods.append('Cooper')
        if all((Te, P, Pc)):
            methods.append('Bier')
        if all((Te, P, Pc)):
            methods.append('Montinsky')
        if all((Te, P, Cpl, kl, sigma, Hvap, rhol, rhog)):
            methods.append('McNelly')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        methods = list_methods()
        if methods == []:
            raise Exception('Insufficient property data for any method.')
        Method = methods[0]

    if Method == 'Stephan-Abdelsalam':
        return Stephan_Abdelsalam(Te=Te, q=q, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               correlation='general', **kwargs)
    elif Method == 'Stephan-Abdelsalam water':
        return Stephan_Abdelsalam(Te=Te, q=q, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               correlation='water', **kwargs)
    elif Method == 'Stephan-Abdelsalam cryogenic':
        return Stephan_Abdelsalam(Te=Te, q=q, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               correlation='cryogenic', **kwargs)
    elif Method == 'HEDH-Taborek':
        return HEDH_Taborek(Te=Te, q=q, P=P, Pc=Pc)
    elif Method == 'Forster-Zuber':
        return Forster_Zuber(Te=Te, q=q, dPsat=dPsat, Cpl=Cpl, kl=kl, mul=mul,
                          sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog, **kwargs)
    elif Method == 'Rohsenow':
        return Rohsenow(Te=Te, q=q, Cpl=Cpl, kl=kl, mul=mul, sigma=sigma, Hvap=Hvap,
                     rhol=rhol, rhog=rhog, **kwargs)
    elif Method == 'Cooper':
        return Cooper(Te=Te, q=q, P=P, Pc=Pc, MW=MW, **kwargs)
    elif Method == 'Bier':
        return Bier(Te=Te, q=q, P=P, Pc=Pc, **kwargs)
    elif Method == 'Montinsky':
        return Montinsky(Te=Te, q=q, P=P, Pc=Pc, **kwargs)
    elif Method == 'McNelly':
        return McNelly(Te=Te, q=q, P=P, Cpl=Cpl, kl=kl, sigma=sigma, Hvap=Hvap,
                    rhol=rhol, rhog=rhog, **kwargs)
    elif Method == 'Gorenflo (1993)':
        return Gorenflo(P=P, q=q, Pc=Pc, Te=Te, CASRN=CAS, **kwargs)
    else:
        raise Exception("Correlation name not recognized; see the "
                        "documentation for the available options.")


### Critical Heat Flux


def Zuber(sigma, Hvap, rhol, rhog, K=0.18):
    r'''Calculates critical heat flux for nucleic boiling of a flat plate
    or other shape as presented in various sources.
    K = pi/24 is believed to be the original [1]_ value for K, but 0.149 is
    now more widely used, a value claimed to be from [2]_ according to [5]_.
    Cao [4]_ lists a value of 0.18 for K. The Wolverine Tube data book also
    lists a value of 0.18, and so it is the default.

    .. math::
        q_c = 0.149H_{vap} \rho_g^{0.5}\left[\sigma g (\rho_L-\rho_g)\right]^{0.25}

    Parameters
    ----------
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    K : float
        Constant []

    Returns
    -------
    q: float
        Critical heat flux [W/m^2]

    Notes
    -----
    No further work is required on this correlation. Multiple sources confirm
    its form.

    Examples
    --------
    Example from [3]_

    >>> Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, K=0.149)
    444307.22304342285
    >>> Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, K=0.18)
    536746.9808578263

    References
    ----------
    .. [1] Zuber N. "On the stability of boiling heat transfer". Trans ASME 1958
        80:711-20.
    .. [2] Lienhard, J.H., and Dhir, V.K., 1973, Extended Hydrodynamic Theory
       of the Peak and Minimum Heat Fluxes, NASA CR-2270.
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [4] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [5] Kreith, Frank, Raj Manglik, and Mark Bohn. Principles of Heat
       Transfer, 7E.Mason, OH: Cengage Learning, 2010.
    '''
    return K*Hvap*rhog**0.5*(g*sigma*(rhol-rhog))**0.25


def Serth_HEDH(D, sigma, Hvap, rhol, rhog):
    r'''Calculates critical heat flux for nucleic boiling of a tube bundle
    according to [2]_, citing [3]_, and using [1]_ as the original form.

    .. math::
        q_c = KH_{vap} \rho_g^{0.5}\left[\sigma g (\rho_L-\rho_g)\right]^{0.25}

    .. math::
        K = 0.123 (R^*)^{-0.25} \text{ for 0.12 < R* < 1.17}

    .. math::
        K = 0.118

    .. math::
        R^* = \frac{D}{2} \left[\frac{g(\rho_L-\rho_G)}{\sigma}\right]^{0.5}

    Parameters
    ----------
    D : float
        Diameter of tubes [m]
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at T, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]

    Returns
    -------
    q: float
        Critical heat flux [W/m^2]

    Notes
    -----
    A further source for this would be nice.

    Examples
    --------
    >>> Serth_HEDH(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    351867.46522901946

    References
    ----------
    .. [1] Zuber N. "On the stability of boiling heat transfer". Trans ASME
       1958 80:711-20.
    .. [2] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    .. [3] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    '''
    R = D/2*(g*(rhol-rhog)/sigma)**0.5
    if 0.12 <= R  <= 1.17:
        K = 0.125*R**-0.25
    else:
        K = 0.118
    return K*Hvap*rhog**0.5*(g*sigma*(rhol-rhog))**0.25


def HEDH_Montinsky(P, Pc):
    r'''Calculates critical heat flux
    in the nucleate boiling regime according to [3]_ as presented in [1]_,
    using an expression modified from [2]_.

    .. math::
        q_c = 367 P_cP_r^{0.35}(1-P_r)^{0.9}

    Parameters
    ----------
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]

    Returns
    -------
    q : float
        Critical heat flux [W/m^2]

    Notes
    -----
    No further work is required.
    Units of Pc are kPa internally.

    Examples
    --------
    Example is from [3]_ and matches to within the error of the algebraic
    manipulation rounding.

    >>> HEDH_Montinsky(P=310.3E3, Pc=2550E3)
    398405.66545181436

    References
    ----------
    .. [1] Schlünder, Ernst U, and International Center for Heat and Mass
       Transfer. Heat Exchanger Design Handbook. Washington:
       Hemisphere Pub. Corp., 1987.
    .. [2] Mostinsky I. L.: "Application of the rule of corresponding states
       for the calculation of heat transfer and critical heat flux,"
       Teploenergetika 4:66, 1963 English Abstr. Br Chem Eng 8(8):586, 1963
    .. [3] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    Pr = P/Pc
    return 367*(Pc/1000.)*Pr**0.35*(1-Pr)**0.9


qmax_boiling_methods = ['Serth-HEDH', 'Zuber', 'HEDH-Montinsky']


def qmax_boiling(rhol=None, rhog=None, sigma=None, Hvap=None, D=None, P=None, 
                 Pc=None, Method=None, AvailableMethods=False):
    r'''This function handles the calculation of nucleate boiling critical
    heat flux and chooses the best method for performing the calculation.
    
    Preferred methods are 'Serth-HEDH' when a tube diameter is specified,
    and 'Zuber' otherwise.

    Parameters
    ----------
    rhol : float, optional
        Density of the liquid [kg/m^3]
    rhog : float, optional
        Density of the produced gas [kg/m^3]
    sigma : float, optional
        Surface tension of liquid [N/m]
    Hvap : float, optional
        Heat of vaporization of the fluid at T, [J/kg]
    D : float, optional
        Diameter of tubes [m]
    P : float, optional
        Saturation pressure of fluid, [Pa]
    Pc : float, optional
        Critical pressure of fluid, [Pa]

    Returns
    -------
    q : float
        Nucleate boiling critical heat flux [W/m^2]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to calculate qmax with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use; one of ('Serth-HEDH', 'Zuber', 
        or 'HEDH-Montinsky')
    AvailableMethods : bool, optional
        If True, function will consider which methods which can be used to
        calculate `qmax` with the given inputs
        
    Examples
    --------
    >>> qmax_boiling(D=0.0127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)
    351867.46522901946
    '''
    def list_methods():
        methods = []
        if all((sigma, Hvap, rhol, rhog, D)):
            methods.append('Serth-HEDH')
        if all((sigma, Hvap, rhol, rhog)):
            methods.append('Zuber')
        if all((P, Pc)):
            methods.append('HEDH-Montinsky')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        methods = list_methods()
        if methods == []:
            raise Exception('Insufficient property or geometry data for any '
                            'method.')
        Method = methods[0]

    if Method == 'Serth-HEDH':
        return Serth_HEDH(D=D, sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method == 'Zuber':
        return Zuber(sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method == 'HEDH-Montinsky':
        return HEDH_Montinsky(P=P, Pc=Pc)
    else:
        raise Exception("Correlation name not recognized; options are "
                        "'Serth-HEDH', 'Zuber' and 'HEDH-Montinsky'")
