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
from math import log, log10

__all__ = ['Rohsenow', 'McNelly', 'Forster_Zuber', 'Montinsky',
'Stephan_Abdelsalam', 'HEDH_Taborek', 'Bier', 'Cooper', 'h_nucleic', 'Zuber',
'Serth_HEDH', 'HEDH_Montinsky', 'qmax_boiling']

def Rohsenow(Te=None, Cpl=None, kl=None, mul=None, sigma=None, Hvap=None,
            rhol=None, rhog=None, Csf=0.013, n=1.7):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    .. math::
        h = {{\mu }_{L}} \Delta H_{vap} \left[ \frac{g( \rho_L-\rho_v)}{\sigma } \right]^{0.5}
        \left[\frac{C_{p,L}\Delta T_e^{2/3}}{C_{sf}\Delta H_{vap} Pr_L^n}\right]^3

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mul : float
        Viscosity of liquid [Pa*s]
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    Csf : float
        Rohsenow coefficient specific to fluid and metal []
    n : float
            Constant, 1 for water, 1.7 (default) for other fluids usually []

    Returns
    -------
    h: float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No further work is required on this correlation. Multiple sources confirm
    its form and rearrangement.

    Examples
    --------
    Q for water at atmospheric pressure on oxidized aluminum, 10.30 P set 8.

    >>> Rohsenow(Te=4.9, Cpl=4217., kl=0.680, mul=2.79E-4, sigma=0.0589,
    ... Hvap=2.257E6, rhol=957.854, rhog=0.595593, Csf=0.011, n=1.26)*4.9
    18245.91080863059

    References
    ----------
    .. [1] Cao, Eduardo. Heat Transfer in Process Engineering.
       McGraw Hill Professional, 2009.
    .. [2] Rohsenow, Warren M. "A Method of Correlating Heat Transfer Data for
       Surface Boiling of Liquids.” Technical Report. Cambridge, Mass. : M.I.T.
       Division of Industrial Cooporation, 1951
    '''
    h = mul*Hvap*(g*(rhol-rhog)/sigma)**0.5*(Cpl*Te**(2/3.)/Csf/Hvap/(Cpl*mul/kl)**n)**3
    return h

#print [Rohsenow(Te=i, Cpl=4180, kl=0.688, mul=2.75E-4, sigma=0.0588, Hvap=2.25E6,
#            rhol=958, rhog=0.597, Csf=0.013, n=1) for i in [4.3, 9.1, 13]]

#print [Rohsenow(Te=5, Cpl=4180, kl=0.688, mul=2.75E-4, sigma=0.0588, Hvap=2.25E6, rhol=958, rhog=0.597)]

def McNelly(Te=None, P=None, Cpl=None, kl=None, sigma=None, Hvap=None,
            rhol=None, rhog=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    .. math::
        h = \left(0.225\left(\frac{\Delta T_e C_{p,l}}{H_{vap}}\right)^{0.69}
        \left(\frac{P k_L}{\sigma}\right)^{0.31}
        \left(\frac{\rho_L}{\rho_V}-1\right)^{0.33}\right)^{1/0.31}

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    P : float
        Saturation pressure of fluid, [Pa]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]

    Returns
    -------
    h: float
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
    h = (0.225*(Te*Cpl/Hvap)**0.69*(P*kl/sigma)**0.31*(rhol/rhog-1.)**0.33
        )**(1./0.31)
    return h


#print [McNelly(Te=4.3, P=101325, Cpl=4180., kl=0.688, sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)] # Water
#print [McNelly(Te=9.1, P=101325., Cpl=4472., kl=0.502, sigma=0.0325, Hvap=1.37E6, rhol=689., rhog=0.843)] # Ammonia


def Forster_Zuber(Te=None, dPSat=None, Cpl=None, kl=None, mul=None, sigma=None,
                  Hvap=None, rhol=None, rhog=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    .. math::
        h = 0.00122\left(\frac{k_L^{0.79} C_{p,l}^{0.45}\rho_L^{0.49}}
        {\sigma^{0.5}\mu_L^{0.29} H_{vap}^{0.24} \rho_V^{0.24}}\right)
        \Delta T_e^{0.24} \Delta P_{sat}^{0.75}

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    dPSat : float
        Difference in Saturation pressure of fluid at Te and T, [Pa]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mul : float
        Viscosity of liquid [Pa*s]
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]

    Returns
    -------
    h: float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Examples have been found in [1]_ and [3]_ and match exactly.

    Examples
    --------
    Water boiling, with excess temperature of 4.3K from [1]_.

    >>> Forster_Zuber(Te=4.3, dPSat=3906*4.3, Cpl=4180., kl=0.688,
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
    h = 0.00122*(kl**0.79*Cpl**0.45*rhol**0.49/sigma**0.5/mul**0.29/Hvap**0.24/rhog**0.24)*Te**0.24*dPSat**0.75
    return h

#print [Forster_Zuber(Te=4.3, dPSat=3906*4.3, Cpl=4180., kl=0.688, mul=0.275E-3,
#                    sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)]
#print [Forster_Zuber(Te=9.1, dPSat=3906*9.1, Cpl=4180., kl=0.688, mul=0.275E-3,
#                    sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)]
#print [Forster_Zuber(Te=13, dPSat=3906*13, Cpl=4180., kl=0.688, mul=0.275E-3,
#                    sigma=0.0588, Hvap=2.25E6, rhol=958., rhog=0.597)]
#print  [Forster_Zuber(Te=16.2, dPSat=106300., Cpl=2730., kl=0.086, mul=156E-6, sigma=.0082,
#                  Hvap=272E3, rhol=567., rhog=18.09)]
#


def Montinsky(Te=None, P=None, Pc=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[1.8(P/P_c)^{0.17} +
        4(P/P_c)^{1.2} + 10(P/P_c)^{10}\right]\right)^{1/0.3}

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]

    Returns
    -------
    h: float
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

    >>> Montinsky(Te=4.3, P=101325., Pc=22048321.0)
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
    h = (0.00417*(Pc/1000.)**0.69*Te**0.7*(1.8*(P/Pc)**0.17 + 4*(P/Pc)**1.2
    +10*(P/Pc)**10))**(1/0.3)
    return h

#print [Montinsky(Te=i, P=101325., Pc=112E5) for i in [4.3, 9.1, 13]]
#print [Montinsky(Te=i, P=101325., Pc=48.9E5) for i in [4.3, 9.1, 13]]
#print [Montinsky(Te=16.2, P=310.3E3, Pc=2550E3)]


_angles_Stephan_Abdelsalam = {'general': 35, 'water': 45, 'hydrocarbon': 35,
'cryogenic': 1, 'refrigerant': 35}

def Stephan_Abdelsalam(Te=None, Tsat=None, Cpl=None, kl=None, mul=None,
     sigma=None, Hvap=None, rhol=None, rhog=None, kw=401., rhow=8.96, Cpw=384,
     angle=None, correlation='general'):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.
    Five variants are possible.

    .. math::
        h = 0.23X_1^{0.674} X_2^{0.35} X_3^{0.371} X_5^{0.297} X_8^{-1.73} k_L/d_B

        X1 = \frac{q D_d}{K_L T_{sat}}

        X2 = \frac{\alpha^2 \rho_L}{\sigma D_d}

        X3 = \frac{C_{p,L} T_{sat} D_d^2}{\alpha^2}

        X4 = \frac{H_{vap} D_d^2}{\alpha^2}

        X5 = \frac{\rho_V}{\rho_L}

        X6 = \frac{C_{p,l} \mu_L}{k_L}

        X7 = \frac{\rho_W C_{p,W} k_W}{\rho_L C_{p,L} k_L}

        X8 = \frac{\rho_L-\rho_V}{\rho_L}

        D_b = 0.0146\theta\sqrt{\frac{2\sigma}{g(\rho_L-\rho_g)}}

    Respectively, the following four correlatoins are for water, hydrocarbons,
    cryogenic fluids, and refrigerants.

    .. math::
        h = 0.246\times 10^7 X1^{0.673} X4^{-1.58} X3^{1.26}X8^{5.22}k_L/d_B

        h = 0.0546 X5^{0.335} X1^{0.67} X8^{-4.33} X4^{0.248}k_L/d_B

        h = 4.82 X1^{0.624} X7^{0.117} X3^{0.374} X4^{-0.329}X5^{0.257} k_L/d_B

        h = 207 X1^{0.745} X5^{0.581} X6^{0.533} k_L/d_B

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    Tsat : float
        Saturation temperature at operating pressure [Pa]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    mul : float
        Viscosity of liquid [Pa*s]
    sigma : float
        Surface tension of liquid [N/m]
    Hvap : float
        Heat of vaporization of the fluid at P, [J/kg]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the produced gas [kg/m^3]
    kw : float
        Thermal conductivity of wall (only for cryogenics) [W/m/K]
    rhow : float
        Density of the wall (only for cryogenics) [kg/m^3]
    Cpw : float
        Heat capacity of wall (only for cryogenics) [J/kg/K]
    angle : float, optional
        Contact angle of bubble with wall [degrees]
    correlation : str, optional
        Any of 'general', 'water', 'hydrocarbon', 'cryogenic', or 'refrigerant'

    Returns
    -------
    h: float
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
    angle = _angles_Stephan_Abdelsalam[correlation]

    db = 0.0146*angle*(2*sigma/g/(rhol-rhog))**0.5
    diffusivity_L = kl/rhol/Cpl

    X1 = db/kl/Tsat*Te
    X2 = diffusivity_L**2*rhol/sigma/db
    X3 = Hvap*db**2/diffusivity_L**2
    X4 = Hvap*db**2/diffusivity_L**2
    X5 = rhog/rhol
    X6 = Cpl*mul/kl
    X7 = rhow*Cpw*kw/(rhol*Cpl*kl)
    X8 = (rhol-rhog)/rhol

    if correlation == 'general':
        h = (0.23*X1**0.674*X2**0.35*X3**0.371*X5**0.297*X8**-1.73*kl/db)**(1/0.326)
    elif correlation == 'water':
        h = (0.246E7*X1**0.673*X4**-1.58*X3**1.26*X8**5.22*kl/db)**(1/0.327)
    elif correlation == 'hydrocarbon':
        h = (0.0546*X5**0.335*X1**0.67*X8**-4.33*X4**0.248*kl/db)**(1/0.33)
    elif correlation == 'cryogenic':
        h = (4.82*X1**0.624*X7**0.117*X3**0.374*X4**-0.329*X5**0.257*kl/db)**(1/0.376)
    else:
        h = (207*X1**0.745*X5**0.581*X6**0.533*kl/db)**(1/0.255)
    return h

#print [Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,
#                          sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, correlation=i) for i in _angles_Stephan_Abdelsalam.keys()]

#print [Stephan_Abdelsalam(Te=16.2, Tsat=437.5, Cpl=2730., kl=0.086, mul=156E-6,
#                  sigma=0.0082, Hvap=272E3, rhol=567, rhog=18.09, angle=35)]

def HEDH_Taborek(Te=None, P=None, Pc=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to Taborek (1986)
    as described in [1]_ and as presented in [2]_. Modification of [3]_.

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[2.1P_r^{0.27} +
        \left(9 + (1-Pr^2)^{-1}\right)P_r^2 \right]\right)^{1/0.3}

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]

    Returns
    -------
    h: float
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
    h = (0.00417*(Pc/1000.)**0.69*Te**0.7*(2.1*Pr**0.27
    + (9 + 1./(1-Pr**2))*Pr**2))**(1/0.3)
    return h


def Bier(Te=None, P=None, Pc=None):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [1]_ .

    .. math::
        h = \left(0.00417P_c^{0.69} \Delta Te^{0.7}\left[0.7 + 2P_r\left(4 +
        \frac{1}{1-P_r}\right)  \right]\right)^{1/0.3}

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]

    Returns
    -------
    h: float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    No examples of this are known. Seems to give very different results than
    other correlations.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3 K from [1]_.

    >>> Bier(4.3, 101325., 22048321.0)
    1290.5349471503353

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    Pr = P/Pc
    h = (0.00417*(Pc/1000.)**0.69*Te**0.7*(0.7 + 2*Pr*(4 + 1/(1-Pr))))**(1/0.3)
    return h

#print [Bier(Te=i, P=101325., Pc=22048321.0) for i in [4.3, 9.1, 13]]
#print [Bier(Te=i, P=101325., Pc=48.9E5) for i in [4.3, 9.1, 13]]

def Cooper(Te=None, P=None, Pc=None, MW=None, Rp=1E-6):
    r'''Calculates heat transfer coefficient for a evaporator operating
    in the nucleate boiling regime according to [2]_ as presented in [1]_.

    .. math::
        h = \left(55\Delta Te^{0.67} \frac{P}{P_c}^{(0.12 - 0.2\log_{10} R_p)}
        (-\log_{10} \frac{P}{P_c})^{-0.55} MW^{-0.5}\right)^{1/0.33}

    Parameters
    ----------
    Te : float
        Excess wal temperature, [K]
    P : float
        Saturation pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    MW : float
        Molecular weight of fluid, [g/mol]
    Rp : float
        Roughness parameter of the surface (1 micrometer default), [m]

    Returns
    -------
    h: float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    Examples 1 and 2 are for water and benzene, from [1]_.
    Roughness parameter is with an old definition. Accordingly, it is
    not used by the h function.
    If unchanged, the roughness parameter's logarithm gives a value of 0.12
    as an exponent of reduced pressure.

    No further work is required.

    Examples
    --------
    Water boiling at 1 atm, with excess temperature of 4.3 K from [1]_.

    >>> Cooper(Te=4.3, P=101325., Pc=22048321.0, MW=18.02)
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
    h = (55*Te**0.67*(P/Pc)**(0.12 - 0.2*log10(Rp))*(
         -log10(P/Pc))**-0.55*MW**-0.5)**(1/0.33)
    return h

#print [Cooper(Te=i, P=101325., Pc=22048321.0, MW=18.02) for i in [4.3, 9.1, 13]]
#print [Cooper(Te=i, P=101325., Pc=48.9E5, MW=78.11184) for i in [4.3, 9.1, 13]]
#print [Cooper(Te=16.2, P=310.3E3, Pc=2550E3, MW=110.37)]

cryogenics = {'132259-10-0': 'Air', '7440-37-1': 'Argon', '630-08-0':
'carbon monoxide', '7782-39-0': 'deuterium', '7782-41-4': 'fluorine',
'7440-59-7': 'helium', '1333-74-0': 'hydrogen', '7439-90-9': 'krypton',
'74-82-8': 'methane', '7440-01-9': 'neon', '7727-37-9': 'nitrogen',
'7782-44-7': 'oxygen', '7440-63-3': 'xenon'}




def h_nucleic(Te=None, Tsat=None, P=None, dPSat=None, dPdT=None,
      Cpl=None, kl=None, mul=None, rhol=None, sigma=None, Hvap=None,
      rhog=None, MW=None, Pc=None, kw=None, rhow=None, Cpw=None, Rp=None,
      CAS=None, AvailableMethods=False, Method=None):
    r'''This function handles choosing which nucleate boiling correlation
    to use, depending on the provided information. Generally this
    is used by a helper class, but can be used directly. Will automatically
    select the correlation to use if none is provided'''
    def list_methods():
        methods = []
        if all((Te, Tsat, Cpl, kl, mul, sigma, Hvap, rhol, rhog)):
            if CAS and CAS == '7732-18-5':
                methods.append('Stephan-Abdelsalam water')
            if CAS and CAS in cryogenics:
                methods.append('Stephan-Abdelsalam cryogenic')
            methods.append('Stephan-Abdelsalam')
        if all((Te, P, Pc)):
            methods.append('HEDH-Taborek')
        if all((Te, dPSat, Cpl, kl, mul, sigma, Hvap, rhol, rhog)):
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
        methods.append('None')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == 'Stephan-Abdelsalam':
        h = Stephan_Abdelsalam(Te=Te, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               kw=kw, rhow=rhow, Cpw=Cpw, correlation='general')
    elif Method == 'Stephan-Abdelsalam water':
        h = Stephan_Abdelsalam(Te=Te, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               kw=kw, rhow=rhow, Cpw=Cpw, correlation='water')
    elif Method == 'Stephan-Abdelsalam cryogenic':
        h = Stephan_Abdelsalam(Te=Te, Tsat=Tsat, Cpl=Cpl, kl=kl, mul=mul,
                               sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog,
                               kw=kw, rhow=rhow, Cpw=Cpw, correlation='cryogenic')
    elif Method == 'HEDH-Taborek':
        h = HEDH_Taborek(Te=Te, P=P, Pc=Pc)
    elif Method == 'Forster-Zuber':
        h = Forster_Zuber(Te=Te, dPSat=dPSat, Cpl=Cpl, kl=kl, mul=mul,
                          sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method == 'Rohsenow':
        h = Rohsenow(Te=Te, Cpl=Cpl, kl=kl, mul=mul, sigma=sigma, Hvap=Hvap,
                     rhol=rhol, rhog=rhog)
    elif Method == 'Cooper':
        h = Cooper(Te=Te, P=P, Pc=Pc, MW=MW)
    elif Method == 'Bier':
        h = Bier(Te=Te, P=P, Pc=Pc)
    elif Method == 'Montinsky':
        h = Montinsky(Te=Te, P=P, Pc=Pc)
    elif Method == 'McNelly':
        h = McNelly(Te=Te, P=P, Cpl=Cpl, kl=kl, sigma=sigma, Hvap=Hvap,
                    rhol=rhol, rhog=rhog)
    elif Method == 'None':
        h = None
    else:
        raise Exception('Failure in in function')
    return h


### Critical Heat Flux
def Zuber(sigma=None, Hvap=None, rhol=None, rhog=None, K=0.18):
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
    q = K*Hvap*rhog**0.5*(g*sigma*(rhol-rhog))**0.25
    return q

#print [Zuber(sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09, K=0.18)]

def Serth_HEDH(D=None, sigma=None, Hvap=None, rhol=None, rhog=None):
    r'''Calculates critical heat flux for nucleic boiling of a tube bundle
    according to [2]_, citing [3]_, and using [1]_ as the original form.

    .. math::
        q_c = KH_{vap} \rho_g^{0.5}\left[\sigma g (\rho_L-\rho_g)\right]^{0.25}

        K = 0.123 (R^*)^{-0.25} \text{ for 0.12 < R* < 1.17}

        K = 0.118

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
    q = K*Hvap*rhog**0.5*(g*sigma*(rhol-rhog))**0.25
    return q

#print [Serth_HEDH(D=0.00127, sigma=8.2E-3, Hvap=272E3, rhol=567, rhog=18.09)]

def HEDH_Montinsky(P=None, Pc=None):
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
    q: float
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
    q = 367*(Pc/1000.)*Pr**0.35*(1-Pr)**0.9
    return q
#print [HEDH_Montinsky(P=310.3E3, Pc=2550E3)]


def qmax_boiling(rhol=None, rhog=None, sigma=None, Hvap=None, D=None, P=None, Pc=None,
         AvailableMethods=False, Method=None):
    r'''This function handles choosing which nucleate boiling critical
    heat flux correlation to use, depending on the provided information.
    Generally this is used by a helper class, but can be used directly. Will
    automatically select the correlation to use if none is provided'''
    def list_methods():
        methods = []
        if all((sigma, Hvap, rhol, rhog, D)):
            methods.append('Serth-HEDH')
        if all((sigma, Hvap, rhol, rhog)):
            methods.append('Zuber')
        if all((P, Pc)):
            methods.append('HEDH-Montinsky')
        methods.append('None')
        return methods

    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == 'Serth-HEDH':
        q = Serth_HEDH(D=D, sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method == 'Zuber':
        q = Zuber(sigma=sigma, Hvap=Hvap, rhol=rhol, rhog=rhog)
    elif Method == 'HEDH-Montinsky':
        q = HEDH_Montinsky(P=P, Pc=Pc)
    elif Method == 'None':
        q = None
    else:
        raise Exception('Failure in in function')
    return q
