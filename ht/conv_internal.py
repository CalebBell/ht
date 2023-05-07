'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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


__all__ = ['laminar_T_const', 'laminar_Q_const',
'laminar_entry_thermal_Hausen', 'laminar_entry_Seider_Tate',
'laminar_entry_Baehr_Stephan', 'turbulent_Dittus_Boelter',
'turbulent_Sieder_Tate', 'turbulent_entry_Hausen', 'turbulent_Colburn',
'turbulent_Drexel_McAdams', 'turbulent_von_Karman', 'turbulent_Prandtl',
'turbulent_Friend_Metzner', 'turbulent_Petukhov_Kirillov_Popov',
'turbulent_Webb', 'turbulent_Sandall', 'turbulent_Gnielinski',
'turbulent_Gnielinski_smooth_1', 'turbulent_Gnielinski_smooth_2',
'turbulent_Churchill_Zajic', 'turbulent_ESDU', 'turbulent_Martinelli',
'turbulent_Nunner', 'turbulent_Dipprey_Sabersky', 'turbulent_Gowen_Smith',
'turbulent_Kawase_Ulbrecht', 'turbulent_Kawase_De', 'turbulent_Bhatti_Shah',
'Nu_conv_internal', 'Nu_conv_internal_methods',

'Morimoto_Hotta', 'helical_turbulent_Nu_Mori_Nakayama',
'helical_turbulent_Nu_Schmidt', 'helical_turbulent_Nu_Xin_Ebadian',
'Nu_laminar_rectangular_Shan_London',
'conv_tube_methods', 'conv_tube_laminar_methods', 'conv_tube_turbulent_methods']

from math import exp, log, tanh

from fluids.friction import LAMINAR_TRANSITION_PIPE, Clamond

### Laminar

def laminar_T_const():
    r'''Returns internal convection Nusselt number for laminar flows
    in pipe according to [1]_, [2]_ and [3]_. Wall temperature is assumed
    constant.
    This is entirely theoretically derived and reproduced experimentally.

    .. math::
        Nu = 3.66

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    This applies only for fully thermally and hydraulically developed and flows.

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry`s Chemical Engineers` Handbook,
       Eighth Edition. New York: McGraw-Hill Education, 2007.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ: Wiley, 2011.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
       Berlin ; New York: Springer, 2010.
    '''
    return 3.66


def laminar_Q_const():
    r'''Returns internal convection Nusselt number for laminar flows
    in pipe according to [1]_, [2]_, and [3]_. Heat flux is assumed constant.
    This is entirely theoretically derived and reproduced experimentally.

    .. math::
        Nu = 4.354

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    This applies only for fully thermally and hydraulically developed and flows.
    Many sources round to 4.36, but [3]_ does not.

    References
    ----------
    .. [1] Green, Don, and Robert Perry. Perry`s Chemical Engineers` Handbook,
       Eighth Edition. New York: McGraw-Hill Education, 2007.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E. Hoboken, NJ: Wiley, 2011.
    .. [3] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd ed. 2010 edition.
        Berlin ; New York: Springer, 2010.
    '''
    return 48/11.

### Laminar - entry region

def laminar_entry_thermal_Hausen(Re, Pr, L, Di):
    r'''Calculates average internal convection Nusselt number for laminar flows
    in pipe during the thermal entry region according to [1]_ as shown in
    [2]_ and cited by [3]_.

    .. math::
        Nu_D=3.66+\frac{0.0668\frac{D}{L}Re_{D}Pr}{1+0.04{(\frac{D}{L}
        Re_{D}Pr)}^{2/3}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    L : float
        Length of pipe [m]
    Di : float
        Diameter of pipe [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    If Pr >> 1, (5 is a common requirement) this equation also applies to flows
    with developing velocity profile.
    As L gets larger, this equation  becomes the constant-temperature Nusselt
    number.

    Examples
    --------
    >>> laminar_entry_thermal_Hausen(Re=100000, Pr=1.1, L=5, Di=.5)
    39.01352358988535

    References
    ----------
    .. [1] Hausen, H. Darstellung des Warmeuberganges in Rohren durch
       verallgeminerte Potenzbeziehungen, Z. Ver deutsch. Ing Beih.
       Verfahrenstech., 4, 91-98, 1943
    .. [2] W. M. Kays. 1953. Numerical Solutions for Laminar Flow Heat Transfer
       in Circular Tubes.
    .. [3] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E.
       Hoboken, NJ: Wiley, 2011.
    '''
    Gz = Di/L*Re*Pr
    return 3.66 + (0.0668*Gz)/(1+0.04*(Gz)**(2/3.))


def laminar_entry_Seider_Tate(Re, Pr, L, Di, mu=None, mu_w=None):
    r'''Calculates average internal convection Nusselt number for laminar flows
    in pipe during the thermal entry region as developed in [1]_, also
    shown in [2]_.

    .. math::
        Nu_D=1.86\left(\frac{D}{L}Re_DPr\right)^{1/3}\left(\frac{\mu_b}
        {\mu_s}\right)^{0.14}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    L : float
        Length of pipe [m]
    Di : float
        Diameter of pipe [m]
    mu : float, optional
        Viscosity of fluid, [Pa*s]
    mu_w : float, optional
        Viscosity of fluid at wall temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Reynolds number should be less than 10000. This should be calculated
    using pipe diameter.
    Prandlt number should be no less than air and no more than liquid metals;
    0.7 < Pr <  16700
    Viscosities should be the bulk and surface properties; they are optional.
    Outside the boundaries, this equation is provides very false results.

    Examples
    --------
    >>> laminar_entry_Seider_Tate(Re=100000, Pr=1.1, L=5, Di=.5)
    41.366029684589265

    References
    ----------
    .. [1] Sieder, E. N., and G. E. Tate. "Heat Transfer and Pressure Drop of
       Liquids in Tubes." Industrial & Engineering Chemistry 28, no. 12
       (December 1, 1936): 1429-35. doi:10.1021/ie50324a027.
    .. [2] Serth, R. W., Process Heat Transfer: Principles,
       Applications and Rules of Thumb. 2E. Amsterdam: Academic Press, 2014.
    '''
    Nu = 1.86*(Di/L*Re*Pr)**(1/3.0)
    if mu_w is not None and mu is not None:
        Nu *= (mu/mu_w)**0.14
    return Nu


def laminar_entry_Baehr_Stephan(Re, Pr, L, Di):
    r'''Calculates average internal convection Nusselt number for laminar flows
    in pipe during the thermal and velocity entry region according to [1]_ as
    shown in [2]_.

    .. math::
        Nu_D=\frac{\frac{3.657}{\tanh[2.264 Gz_D^{-1/3}+1.7Gz_D^{-2/3}]}
        +0.0499Gz_D\tanh(Gz_D^{-1})}{\tanh(2.432Pr^{1/6}Gz_D^{-1/6})}

    .. math::
        Gz = \frac{D}{L}Re_D Pr

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    L : float
        Length of pipe [m]
    Di : float
        Diameter of pipe [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    As L gets larger, this equation becomes the constant-temperature Nusselt
    number.

    Examples
    --------
    >>> laminar_entry_Baehr_Stephan(Re=100000, Pr=1.1, L=5, Di=.5)
    72.65402046550976

    References
    ----------
    .. [1] Baehr, Hans Dieter, and Karl Stephan. Heat and Mass Transfer.
       Springer, 2013.
    .. [2] Bergman, Theodore L., Adrienne S. Lavine, Frank P. Incropera, and
       David P. DeWitt. Introduction to Heat Transfer. 6E.
       Hoboken, NJ: Wiley, 2011.
    '''
    Gz = Di/L*Re*Pr
    return (3.657/tanh(2.264*Gz**(-1/3.)+ 1.7*Gz**(-2/3.0))
            + 0.0499*Gz*tanh(1./Gz))/tanh(2.432*Pr**(1/6.0)*Gz**(-1/6.0))


### Turbulent - Equations with more complicated options
def turbulent_Dittus_Boelter(Re, Pr, heating=True, revised=True):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [1]_, and [2]_, a reprint of [3]_.

    .. math::
        Nu = m*Re_D^{4/5}Pr^n

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    heating : bool
        Indicates if the process is heating or cooling, optional
    revised : bool
        Indicates if revised coefficients should be used or not

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    The revised coefficient is m = 0.023.
    The original form of Dittus-Boelter has a linear coefficient of 0.0243
    for heating and 0.0265 for cooling. These are sometimes rounded to 0.024
    and 0.026 respectively.
    The default, heating, provides n = 0.4. Cooling makes n 0.3.

    0.6 ≤ Pr ≤  160
    Re_{D} ≥ 10000
    L/D ≥ 10

    Examples
    --------
    >>> turbulent_Dittus_Boelter(Re=1E5, Pr=1.2)
    247.40036409449127
    >>> turbulent_Dittus_Boelter(Re=1E5, Pr=1.2, heating=False)
    242.9305927410295

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Dittus, F. W., and L. M. K. Boelter. "Heat Transfer in Automobile
       Radiators of the Tubular Type." International Communications in Heat
       and Mass Transfer 12, no. 1 (January 1985): 3-22.
       doi:10.1016/0735-1933(85)90003-X
    .. [3] Dittus, F. W., and L. M. K. Boelter, University of California
       Publications in Engineering, Vol. 2, No. 13, pp. 443-461, October 17,
       1930.
    '''
    m = 0.023
    if heating:
        power = 0.4
    else:
        power = 0.3

    if heating and not revised:
        m = 0.0243
    elif not heating and not revised:
        m = 0.0265
    else:
        m = 0.023
    return m*Re**0.8*Pr**power


def turbulent_Sieder_Tate(Re, Pr, mu=None, mu_w=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [1]_ and supposedly [2]_.

    .. math::
        Nu = 0.027Re^{4/5}Pr^{1/3}\left(\frac{\mu}{\mu_s}\right)^{0.14}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    mu : float
        Viscosity of fluid, [Pa*s]
    mu_w : float
        Viscosity of fluid at wall temperature, [Pa*s]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    A linear coefficient of 0.023 is often listed with this equation. The
    source of the discrepancy is not known. The equation is not present in the
    original paper, but is nevertheless the source usually cited for it.

    Examples
    --------
    >>> turbulent_Sieder_Tate(Re=1E5, Pr=1.2)
    286.9178136793052
    >>> turbulent_Sieder_Tate(Re=1E5, Pr=1.2, mu=0.01, mu_w=0.067)
    219.84016455766044

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Sieder, E. N., and G. E. Tate. "Heat Transfer and Pressure Drop of
       Liquids in Tubes." Industrial & Engineering Chemistry 28, no. 12
       (December 1, 1936): 1429-35. doi:10.1021/ie50324a027.
    '''
    Nu = 0.027*Re**0.8*Pr**(1/3.)
    if mu_w is not None and mu is not None:
        Nu *= (mu/mu_w)**0.14
    return Nu


def turbulent_entry_Hausen(Re, Pr, Di, x):
    r'''Calculates internal convection Nusselt number for the entry region
    of a turbulent flow in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = 0.037(Re^{0.75} - 180)Pr^{0.42}[1+(x/D)^{-2/3}]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    Di : float
        Inside diameter of pipe, [m]
    x : float
        Length inside of pipe for calculation, [m]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.7 < Pr ≤ 3  and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_entry_Hausen(Re=1E5, Pr=1.2, Di=0.154, x=0.05)
    677.7228275901755

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] H. Hausen, "Neue Gleichungen fÜr die Wärmeübertragung bei freier
       oder erzwungener Stromung,"Allg. Warmetchn., (9): 75-79, 1959.
    '''
    return 0.037*(Re**0.75 - 180)*Pr**0.42*(1 + (x/Di)**(-2/3.))


### Regular correlations, Re, Pr and fd only


def turbulent_Colburn(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = 0.023Re^{0.8}Pr^{1/3}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr < 3  and 10^4 < Re < 10^5.

    Examples
    --------
    >>> turbulent_Colburn(Re=1E5, Pr=1.2)
    244.41147091200068

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Colburn, Allan P. "A Method of Correlating Forced Convection
       Heat-Transfer Data and a Comparison with Fluid Friction." International
       Journal of Heat and Mass Transfer 7, no. 12 (December 1964): 1359-84.
       doi:10.1016/0017-9310(64)90125-5.
    '''
    return 0.023*Re**0.8*Pr**(1/3.)


def turbulent_Drexel_McAdams(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = 0.021Re^{0.8}Pr^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is Pr ≤ 0.7 and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Drexel_McAdams(Re=1E5, Pr=0.6)
    171.19055301724387

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Drexel, Rober E., and William H. Mcadams. "Heat-Transfer
       Coefficients for Air Flowing in Round Tubes, in Rectangular Ducts, and
       around Finned Cylinders," February 1, 1945.
       http://ntrs.nasa.gov/search.jsp?R=19930090924.
    '''
    return 0.021*Re**0.8*Pr**(0.4)


def turbulent_von_Karman(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)Re Pr}{1 + 5(f/8)^{0.5}\left[Pr-1+\ln\left(\frac{5Pr+1}
        {6}\right)\right]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 ≤ Pr ≤ 3  and 10^4 ≤ Re ≤ 10^5.

    Examples
    --------
    >>> turbulent_von_Karman(Re=1E5, Pr=1.2, fd=0.0185)
    255.7243541243272

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] T. von Karman, "The Analogy Between Fluid Friction and Heat
       Transfer," Trans. ASME, (61):705-710,1939.
    '''
    return (fd/8.0*Re*Pr/(1.0 + 5.0*(fd/8.0)**0.5
                          *(Pr - 1.0 + log((5.0*Pr + 1.0)/6.))))


def turbulent_Prandtl(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{1 + 8.7(f/8)^{0.5}(Pr-1)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ 0.5 ≤ Pr ≤ 5 and 10^4 ≤ Re ≤ 5*10^6

    Examples
    --------
    >>> turbulent_Prandtl(Re=1E5, Pr=1.2, fd=0.0185)
    256.073339689557

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] L. Prandt, Fuhrrer durch die Stomungslehre, Vieweg, Braunschweig,
       p. 359, 1944.
    '''
    return (fd/8.)*Re*Pr/(1.0 + 8.7*(fd/8.)**0.5*(Pr - 1.0))


def turbulent_Friend_Metzner(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{1.2 + 11.8(f/8)^{0.5}(Pr-1)Pr^{-1/3}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ 50 < Pr ≤ 600  and 5*10^4 ≤ Re ≤ 5*10^6.
    The extreme limits on range should be considered!

    Examples
    --------
    >>> turbulent_Friend_Metzner(Re=1E5, Pr=100., fd=0.0185)
    1738.3356262055322

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Friend, W. L., and A. B. Metzner. “Turbulent Heat Transfer inside
       Tubes and the Analogy among Heat, Mass, and Momentum Transfer.” AIChE
       Journal 4, no. 4 (December 1, 1958): 393-402. doi:10.1002/aic.690040404.
    '''
    return (fd/8.)*Re*Pr/(1.2 + 11.8*(fd/8.)**0.5*(Pr - 1.)*Pr**(-1/3.))


def turbulent_Petukhov_Kirillov_Popov(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ and [3]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{C+12.7(f/8)^{1/2}(Pr^{2/3}-1)}\\
        C = 1.07 + 900/Re - [0.63/(1+10Pr)]

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 10^6  and 4000 ≤ Re ≤ 5*10^6

    Examples
    --------
    >>> turbulent_Petukhov_Kirillov_Popov(Re=1E5, Pr=1.2, fd=0.0185)
    250.11935088905105

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] B. S. Petukhov, and V. V. Kirillov, "The Problem of Heat Exchange
       in the Turbulent Flow of Liquids in Tubes," (Russian) Teploenergetika,
       (4): 63-68, 1958
    .. [3] B. S. Petukhov and V. N. Popov, "Theoretical Calculation of Heat
       Exchange in Turbulent Flow in Tubes of an Incompressible Fluidwith
       Variable Physical Properties," High Temp., (111): 69-83, 1963.
    '''
    C = 1.07 + 900./Re - (0.63/(1. + 10.*Pr))
    return (fd/8.)*Re*Pr/(C + 12.7*(fd/8.)**0.5*(Pr**(2/3.) - 1.))


def turbulent_Webb(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{1.07 + 9(f/8)^{0.5}(Pr-1)Pr^{1/4}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 100  and 10^4 ≤ Re ≤ 5*10^6

    Examples
    --------
    >>> turbulent_Webb(Re=1E5, Pr=1.2, fd=0.0185)
    239.10130376815872

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Webb, Dr R. L. “A Critical Evaluation of Analytical Solutions and
       Reynolds Analogy Equations for Turbulent Heat and Mass Transfer in
       Smooth Tubes.” Wärme - Und Stoffübertragung 4, no. 4
       (December 1, 1971): 197-204. doi:10.1007/BF01002474.
    '''
    return (fd/8.)*Re*Pr/(1.07 + 9.*(fd/8.)**0.5*(Pr - 1.)*Pr**0.25)


def turbulent_Sandall(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_.

    .. math::
        Nu = \frac{(f/8)RePr}{12.48Pr^{2/3} - 7.853Pr^{1/3} + 3.613\ln Pr + 5.8 + C}\\
        C = 2.78\ln((f/8)^{0.5} Re/45)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5< Pr ≤ 2000  and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Sandall(Re=1E5, Pr=1.2, fd=0.0185)
    229.0514352970239

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Sandall, O. C., O. T. Hanna, and P. R. Mazet. “A New Theoretical
       Formula for Turbulent Heat and Mass Transfer with Gases or Liquids in
       Tube Flow.” The Canadian Journal of Chemical Engineering 58, no. 4
       (August 1, 1980): 443-47. doi:10.1002/cjce.5450580404.
    '''
    C = 2.78*log((fd/8.)**0.5*Re/45.)
    return (fd/8.)**0.5*Re*Pr/(12.48*Pr**(2/3.) - 7.853*Pr**(1/3.)
                               + 3.613*log(Pr) + 5.8 + C)


def turbulent_Gnielinski(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_. This is the most recent general
    equation, and is strongly recommended.

    .. math::
        Nu = \frac{(f/8)(Re-1000)Pr}{1+12.7(f/8)^{1/2}(Pr^{2/3}-1)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 2000  and 2300 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Gnielinski(Re=1E5, Pr=1.2, fd=0.0185)
    254.62682749359632

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gnielinski, V. (1976). New Equation for Heat and Mass Transfer in
       Turbulent Pipe and Channel Flow, International Chemical Engineering,
       Vol. 16, pp. 359-368.
    '''
    return (fd/8.)*(Re - 1E3)*Pr/(1. + 12.7*(fd/8.)**0.5*(Pr**(2/3.) - 1.))


def turbulent_Gnielinski_smooth_1(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_. This is a simplified case assuming
    smooth pipe.

    .. math::
        Nu = 0.0214(Re^{0.8}-100)Pr^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 0.5 < Pr ≤ 1.5  and 10^4 ≤ Re ≤ 5*10^6.

    Examples
    --------
    >>> turbulent_Gnielinski_smooth_1(Re=1E5, Pr=1.2)
    227.88800494373442

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gnielinski, V. (1976). New Equation for Heat and Mass Transfer in
       Turbulent Pipe and Channel Flow, International Chemical Engineering,
       Vol. 16, pp. 359-368.
    '''
    return 0.0214*(Re**0.8 - 100.)*Pr**0.4


def turbulent_Gnielinski_smooth_2(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as in [1]_. This is a simplified case assuming
    smooth pipe.

    .. math::
        Nu = 0.012(Re^{0.87}-280)Pr^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Range according to [1]_ is 1.5 < Pr ≤ 500 and 3*10^3 ≤ Re ≤ 10^6.

    Examples
    --------
    >>> turbulent_Gnielinski_smooth_2(Re=1E5, Pr=7.)
    577.7692524513449

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gnielinski, V. (1976). New Equation for Heat and Mass Transfer in
       Turbulent Pipe and Channel Flow, International Chemical Engineering,
       Vol. 16, pp. 359-368.
    '''
    return 0.012*(Re**0.87 - 280.)*Pr**0.4


def turbulent_Churchill_Zajic(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as developed in [1]_. Has yet to obtain
    popularity.

    .. math::
        Nu = \left\{\left(\frac{Pr_T}{Pr}\right)\frac{1}{Nu_{di}} +
        \left[1-\left(\frac{Pr_T}{Pr}\right)^{2/3}\right]\frac{1}{Nu_{D\infty}}
        \right\}^{-1}

    .. math::
        Nu_{di} = \frac{Re(f/8)}{1 + 145(8/f)^{-5/4}}

    .. math::
        Nu_{D\infty} = 0.07343Re\left(\frac{Pr}{Pr_T}\right)^{1/3}(f/8)^{0.5}

    .. math::
        Pr_T = 0.85 + \frac{0.015}{Pr}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    No restrictions on range. This is equation is developed with more
    theoretical work than others.

    Examples
    --------
    >>> turbulent_Churchill_Zajic(Re=1E5, Pr=1.2, fd=0.0185)
    260.5564907817961

    References
    ----------
    .. [1] Churchill, Stuart W., and Stefan C. Zajic. “Prediction of Fully
       Developed Turbulent Convection with Minimal Explicit Empiricism.”
       AIChE Journal 48, no. 5 (May 1, 2002): 927-40. doi:10.1002/aic.690480503.
    .. [2] Plawsky, Joel L. Transport Phenomena Fundamentals, Third Edition.
       CRC Press, 2014.
    '''
    Pr_T = 0.85 + 0.015/Pr
    Nu_di = Re*(fd/8.)/(1. + 145*(8./fd)**(-1.25))
    Nu_dinf = 0.07343*Re*(Pr/Pr_T)**(1./3.0)*(fd/8.)**0.5
    return 1./(Pr_T/Pr/Nu_di + (1. - (Pr_T/Pr)**(2/3.))/Nu_dinf)


def turbulent_ESDU(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to the ESDU as shown in [1]_.

    .. math::
        Nu = 0.0225Re^{0.795}Pr^{0.495}\exp(-0.0225\ln(Pr)^2)

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    4000 < Re < 1E6, 0.3 < Pr < 3000 and L/D > 60.
    This equation has not been checked. It was developed by a commercial group.
    This function is a small part of a much larger series of expressions
    accounting for many factors.

    Examples
    --------
    >>> turbulent_ESDU(Re=1E5, Pr=1.2)
    232.3017143430645

    References
    ----------
    .. [1] Hewitt, G. L. Shires, T. Reg Bott G. F., George L. Shires, and
       T. R. Bott. Process Heat Transfer. 1E. Boca Raton: CRC Press, 1994.
    '''
    return 0.0225*Re**0.795*Pr**0.495*exp(-0.0225*log(Pr)**2)

### Correlations for 'rough' turbulent pipe

def turbulent_Martinelli(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu  = \frac{RePr(f/8)^{0.5}}{5[Pr + \ln(1+5Pr) + 0.5\ln(Re(f/8)^{0.5}/60)]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    No range is given for this equation. Liquid metals are probably its only
    applicability.

    Examples
    --------
    >>> turbulent_Martinelli(Re=1E5, Pr=100., fd=0.0185)
    887.1710686396347

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Martinelli, R. C. (1947). "Heat transfer to molten metals".
       Trans. ASME, 69, 947-959.
    '''
    return Re*Pr*(fd/8.)**0.5/5/(Pr + log(1. + 5.*Pr) + 0.5*log(Re*(fd/8.)**0.5/60.))


def turbulent_Nunner(Re, Pr, fd, fd_smooth):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = \frac{RePr(f/8)}{1 + 1.5Re^{-1/8}Pr^{-1/6}[Pr(f/f_s)-1]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]
    fd_smooth : float
        Darcy friction factor of a smooth pipe [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    Valid for Pr ≅ 0.7; bad results for Pr > 1.

    Examples
    --------
    >>> turbulent_Nunner(Re=1E5, Pr=0.7, fd=0.0185, fd_smooth=0.005)
    101.15841010919947

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] W. Nunner, "Warmeiibergang und Druckabfall in Rauhen Rohren,"
       VDI-Forschungsheft 445, ser. B,(22): 5-39, 1956
    '''
    return Re*Pr*fd/8./(1 + 1.5*Re**-0.125*Pr**(-1/6.)*(Pr*fd/fd_smooth - 1.))


def turbulent_Dipprey_Sabersky(Re, Pr, fd, eD):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = \frac{RePr(f/8)}{1 + (f/8)^{0.5}[5.19Re_\epsilon^{0.2} Pr^{0.44} - 8.48]}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    According to [1]_, the limits are:
    1.2 ≤ Pr ≤ 5.94 and 1.4*10^4 ≤ Re ≤ 5E5 and 0.0024 ≤ eD ≤ 0.049.

    Examples
    --------
    >>> turbulent_Dipprey_Sabersky(Re=1E5, Pr=1.2, fd=0.0185, eD=1E-3)
    288.33365198566656

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Dipprey, D. F., and R. H. Sabersky. “Heat and Momentum Transfer in
       Smooth and Rough Tubes at Various Prandtl Numbers.” International
       Journal of Heat and Mass Transfer 6, no. 5 (May 1963): 329-53.
       doi:10.1016/0017-9310(63)90097-8
    '''
    Re_e = Re*eD*(fd/8.)**0.5
    return Re*Pr*fd/8./(1 + (fd/8.)**0.5*(5.19*Re_e**0.2*Pr**0.44 - 8.48))


def turbulent_Gowen_Smith(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = \frac{Re Pr (f/8)^{0.5}} {4.5 + [0.155(Re(f/8)^{0.5})^{0.54}
        + (8/f)^{0.5}]Pr^{0.5}}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    0.7 ≤ Pr ≤ 14.3 and 10^4 ≤ Re ≤ 5E4 and 0.0021 ≤ eD ≤ 0.095

    Examples
    --------
    >>> turbulent_Gowen_Smith(Re=1E5, Pr=1.2, fd=0.0185)
    131.72530453824106

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Gowen, R. A., and J. W. Smith. “Turbulent Heat Transfer from Smooth
       and Rough Surfaces.” International Journal of Heat and Mass Transfer 11,
       no. 11 (November 1968): 1657-74. doi:10.1016/0017-9310(68)90046-X.
    '''
    return Re*Pr*(fd/8.)**0.5/(4.5 + (0.155*(Re*(fd/8.)**0.5)**0.54 + (8./fd)**0.5)*Pr**0.5)


def turbulent_Kawase_Ulbrecht(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = 0.0523RePr^{0.5}(f/4)^{0.5}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    No limits are provided.

    Examples
    --------
    >>> turbulent_Kawase_Ulbrecht(Re=1E5, Pr=1.2, fd=0.0185)
    389.6262247333975

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Kawase, Yoshinori, and Jaromir J. Ulbrecht. “Turbulent Heat and Mass
       Transfer in Dilute Polymer Solutions.” Chemical Engineering Science 37,
       no. 7 (1982): 1039-46. doi:10.1016/0009-2509(82)80134-6.
    '''
    return 0.0523*Re*Pr**0.5*(fd/4.)**0.5


def turbulent_Kawase_De(Re, Pr, fd):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_.

    .. math::
        Nu = 0.0471 RePr^{0.5}(f/4)^{0.5}(1.11 + 0.44Pr^{-1/3} - 0.7Pr^{-1/6})

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    5.1 ≤ Pr ≤ 390 and 5000 ≤ Re ≤ 5E5 and 0.0024 ≤ eD ≤ 0.165.

    Examples
    --------
    >>> turbulent_Kawase_De(Re=1E5, Pr=1.2, fd=0.0185)
    296.5019733271324

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] Kawase, Yoshinori, and Addie De. “Turbulent Heat and Mass Transfer
       in Newtonian and Dilute Polymer Solutions Flowing through Rough Pipes.”
       International Journal of Heat and Mass Transfer 27, no. 1
       (January 1984): 140-42. doi:10.1016/0017-9310(84)90246-1.
    '''
    return 0.0471*Re*Pr**0.5*(fd/4.)**0.5*(1.11 + 0.44*Pr**(-1/3.) - 0.7*Pr**(-1/6.))


def turbulent_Bhatti_Shah(Re, Pr, fd, eD):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in pipe according to [2]_ as shown in [1]_. The most widely used rough
    pipe turbulent correlation.

    .. math::
        Nu_D = \frac{(f/8)Re_DPr}{1+\sqrt{f/8}(4.5Re_{\epsilon}^{0.2}Pr^{0.5}-8.48)}

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    fd : float
        Darcy friction factor [-]
    eD : float
        Relative roughness, [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Notes
    -----
    According to [1]_, the limits are:
    0.5 ≤ Pr ≤  10
    0.002 ≤ ε/D ≤  0.05
    10,000 ≤ Re_{D}
    Another correlation is listed in this equation, with a wider variety
    of validity.

    Examples
    --------
    >>> turbulent_Bhatti_Shah(Re=1E5, Pr=1.2, fd=0.0185, eD=1E-3)
    302.7037617414273

    References
    ----------
    .. [1] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    .. [2] M. S. Bhatti and R. K. Shah. Turbulent and transition flow
       convective heat transfer in ducts. In S. Kakaç, R. K. Shah, and W.
       Aung, editors, Handbook of Single-Phase Convective Heat Transfer,
       chapter 4. Wiley-Interscience, New York, 1987.
    '''
    Re_e = Re*eD*(fd/8.)**0.5
    return Re*Pr*fd/8./(1 + (fd/8.)**0.5*(4.5*Re_e**0.2*Pr**0.5 - 8.48))


conv_tube_laminar_methods = {
    'Laminar - constant T': (laminar_T_const, ()),
    'Laminar - constant Q': (laminar_Q_const, ()),
    'Baehr-Stephan laminar thermal/velocity entry': (laminar_entry_thermal_Hausen, ('Re', 'Pr', 'L', 'Di')),
     'Hausen laminar thermal entry': (laminar_entry_Seider_Tate, ('Re', 'Pr', 'L', 'Di')),
    'Seider-Tate laminar thermal entry': (laminar_entry_Baehr_Stephan, ('Re', 'Pr', 'L', 'Di')),
}

conv_tube_turbulent_methods = {
    'Churchill-Zajic': (turbulent_Churchill_Zajic, ('Re', 'Pr', 'fd')),
    'Petukhov-Kirillov-Popov': (turbulent_Petukhov_Kirillov_Popov, ('Re', 'Pr', 'fd')),
    'Gnielinski': (turbulent_Gnielinski, ('Re', 'Pr', 'fd')),
    'Sandall': (turbulent_Sandall, ('Re', 'Pr', 'fd')),
    'Webb': (turbulent_Webb, ('Re', 'Pr', 'fd')),
    'Friend-Metzner': (turbulent_Friend_Metzner, ('Re', 'Pr', 'fd')),
    'Prandtl': (turbulent_Prandtl, ('Re', 'Pr', 'fd')),
    'von-Karman': (turbulent_von_Karman, ('Re', 'Pr', 'fd')),
    'Martinelli': (turbulent_Martinelli, ('Re', 'Pr', 'fd')),
    'Gowen-Smith': (turbulent_Gowen_Smith, ('Re', 'Pr', 'fd')),
    'Kawase-Ulbrecht': (turbulent_Kawase_Ulbrecht, ('Re', 'Pr', 'fd')),
    'Kawase-De': (turbulent_Kawase_De, ('Re', 'Pr', 'fd')),

    'Dittus-Boelter': (turbulent_Dittus_Boelter, ('Re', 'Pr')),
    'Sieder-Tate': (turbulent_Sieder_Tate, ('Re', 'Pr')),
    'Drexel-McAdams': (turbulent_Drexel_McAdams, ('Re', 'Pr')),
    'Colburn': (turbulent_Colburn, ('Re', 'Pr')),
    'ESDU': (turbulent_ESDU, ('Re', 'Pr')),
    'Gnielinski smooth low Pr': (turbulent_Gnielinski_smooth_1, ('Re', 'Pr')),
    'Gnielinski smooth high Pr': (turbulent_Gnielinski_smooth_2, ('Re', 'Pr')),

    'Hausen': (turbulent_entry_Hausen, ('Re', 'Pr', 'Di', 'x')),
    'Bhatti-Shah': (turbulent_Bhatti_Shah, ('Re', 'Pr', 'fd', 'eD')),
    'Dipprey-Sabersky': (turbulent_Dipprey_Sabersky, ('Re', 'Pr', 'fd', 'eD')),
    'Nunner': (turbulent_Nunner, ('Re', 'Pr', 'fd', 'fd_smooth')),
}

conv_tube_methods = conv_tube_laminar_methods.copy()
conv_tube_methods.update(conv_tube_turbulent_methods)
conv_tube_methods_list = list(conv_tube_methods.keys())

def Nu_conv_internal_methods(Re, Pr, eD=0, Di=None, x=None, fd=None,
                             check_ranges=True):
    r'''This function returns a list of correlation names for the calculation
    of heat transfer coefficient for internal convection inside a circular pipe.

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    eD : float, optional
        Relative roughness, [-]
    Di : float, optional
        Inside diameter of pipe, [m]
    x : float, optional
        Length inside of pipe for calculation, [m]
    fd : float, optoinal
        Darcy friction factor [-]
    check_ranges : bool, optional
        Whether or not to return only correlations suitable for the provided
        data, [-]

    Returns
    -------
    methods : list
        List of methods which can be used to calculate `Nu` with the given inputs

    Examples
    --------
    Turbulent example

    >>> Nu_conv_internal_methods(Re=1E5, Pr=.7)[0]
    'Churchill-Zajic'

    Entry length - laminar example

    >>> Nu_conv_internal_methods(Re=1E2, Pr=.7, x=.01, Di=.1)[0]
    'Baehr-Stephan laminar thermal/velocity entry'
    '''
    methods = []
    if Re < LAMINAR_TRANSITION_PIPE or not check_ranges:
        # Laminar!
        if (Re is not None and Pr is not None and x is not None and Di is not None):
            methods.append('Baehr-Stephan laminar thermal/velocity entry')
            methods.append('Hausen laminar thermal entry')
            methods.append('Seider-Tate laminar thermal entry')

        methods.append('Laminar - constant T')
        methods.append('Laminar - constant Q')
    if Re >= LAMINAR_TRANSITION_PIPE or not check_ranges:
        if (Re is not None and Pr is not None and Pr < 0.03) or not check_ranges:
            # Liquid metals
            methods.append('Martinelli')
        if (Re is not None and Pr is not None and x is not None and Di is not None) or not check_ranges:
            methods.append('Hausen')
        if (Re is not None and Pr is not None and (eD is not None or fd is not None)) or not check_ranges:
            # handle correlations with roughness
            methods.append('Churchill-Zajic')
            methods.append('Petukhov-Kirillov-Popov')
            methods.append('Gnielinski')
            methods.append('Bhatti-Shah')
            methods.append('Dipprey-Sabersky')
            methods.append('Sandall')
            methods.append('Webb')
            methods.append('Friend-Metzner')
            methods.append('Prandtl')
            methods.append('von-Karman')
            methods.append('Gowen-Smith')
            methods.append('Kawase-Ulbrecht')
            methods.append('Kawase-De')
            methods.append('Nunner')
        if (Re is not None and Pr is not None) or not check_ranges:
            methods.append('Dittus-Boelter')
            methods.append('Sieder-Tate')
            methods.append('Drexel-McAdams')
            methods.append('Colburn')
            methods.append('ESDU')
            methods.append('Gnielinski smooth low Pr') # 1
            methods.append('Gnielinski smooth high Pr') # 2
    return methods

def Nu_conv_internal(Re, Pr, eD=0.0, Di=None, x=None, fd=None, Method=None):
    r'''This function calculates the heat transfer coefficient for internal
    convection inside a circular pipe.

    Requires at a minimum a flow's Reynolds and Prandtl numbers `Re` and `Pr`.
    Relative roughness `eD` can be specified to include the enhancement of heat
    transfer from the added turbulence.

    For laminar flow, thermally and hydraulically developing flow is supported
    with the pipe diameter `Di` and distance `x` is provided.

    If no correlation's name is provided as `Method`, the most accurate
    applicable correlation is selected.

    * If laminar, `x` and `Di` provided:  'Baehr-Stephan laminar thermal/velocity entry'
    * Otherwise if laminar, no entry information provided: 'Laminar - constant T' (Nu = 3.66)
    * If turbulent and `Pr` < 0.03: 'Martinelli'
    * If turbulent, `x` and `Di` provided: 'Hausen'
    * Otherwise if turbulent: 'Churchill-Zajic'

    Parameters
    ----------
    Re : float
        Reynolds number, [-]
    Pr : float
        Prandtl number, [-]
    eD : float, optional
        Relative roughness, [-]
    Di : float, optional
        Inside diameter of pipe, [m]
    x : float, optional
        Length inside of pipe for calculation, [m]
    fd : float, optoinal
        Darcy friction factor [-]

    Returns
    -------
    Nu : float
        Nusselt number, [-]

    Other Parameters
    ----------------
    Method : string, optional
        A string of the function name to use, as in the dictionary
        vertical_cylinder_correlations

    Examples
    --------
    Turbulent example

    >>> Nu_conv_internal(Re=1E5, Pr=.7)
    183.71057902604906

    Entry length - laminar example

    >>> Nu_conv_internal(Re=1E2, Pr=.7, x=.01, Di=.1)
    14.91799128769779
    '''
    if Method is None:
        Method2 = Nu_conv_internal_methods(Re=Re, Pr=Pr, eD=eD, Di=Di, x=x, fd=fd, check_ranges=True)[0]
    else:
        Method2 = Method

    L = x
    if eD is not None and fd is None:
        fd = Clamond(Re=Re, eD=eD)

    if Method2 == "Laminar - constant T":
        return laminar_T_const()
    elif Method2 == "Laminar - constant Q":
        return laminar_Q_const()
    elif Method2 == "Baehr-Stephan laminar thermal/velocity entry":
        return laminar_entry_thermal_Hausen(Re=Re, Pr=Pr, L=L, Di=Di)
    elif Method2 == "Hausen laminar thermal entry":
        return laminar_entry_Seider_Tate(Re=Re, Pr=Pr, L=L, Di=Di)
    elif Method2 == "Seider-Tate laminar thermal entry":
        return laminar_entry_Baehr_Stephan(Re=Re, Pr=Pr, L=L, Di=Di)
    elif Method2 == "Churchill-Zajic":
        return turbulent_Churchill_Zajic(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Petukhov-Kirillov-Popov":
        return turbulent_Petukhov_Kirillov_Popov(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Gnielinski":
        return turbulent_Gnielinski(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Sandall":
        return turbulent_Sandall(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Webb":
        return turbulent_Webb(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Friend-Metzner":
        return turbulent_Friend_Metzner(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Prandtl":
        return turbulent_Prandtl(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "von-Karman":
        return turbulent_von_Karman(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Martinelli":
        return turbulent_Martinelli(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Gowen-Smith":
        return turbulent_Gowen_Smith(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Kawase-Ulbrecht":
        return turbulent_Kawase_Ulbrecht(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Kawase-De":
        return turbulent_Kawase_De(Re=Re, Pr=Pr, fd=fd)
    elif Method2 == "Dittus-Boelter":
        return turbulent_Dittus_Boelter(Re=Re, Pr=Pr)
    elif Method2 == "Sieder-Tate":
        return turbulent_Sieder_Tate(Re=Re, Pr=Pr)
    elif Method2 == "Drexel-McAdams":
        return turbulent_Drexel_McAdams(Re=Re, Pr=Pr)
    elif Method2 == "Colburn":
        return turbulent_Colburn(Re=Re, Pr=Pr)
    elif Method2 == "ESDU":
        return turbulent_ESDU(Re=Re, Pr=Pr)
    elif Method2 == "Gnielinski smooth low Pr":
        return turbulent_Gnielinski_smooth_1(Re=Re, Pr=Pr)
    elif Method2 == "Gnielinski smooth high Pr":
        return turbulent_Gnielinski_smooth_2(Re=Re, Pr=Pr)
    elif Method2 == "Hausen":
        return turbulent_entry_Hausen(Re=Re, Pr=Pr, Di=Di, x=x)
    elif Method2 == "Bhatti-Shah":
        return turbulent_Bhatti_Shah(Re=Re, Pr=Pr, fd=fd, eD=eD)
    elif Method2 == "Dipprey-Sabersky":
        return turbulent_Dipprey_Sabersky(Re=Re, Pr=Pr, fd=fd, eD=eD)
    elif Method2 == "Nunner":
        fd_smooth = Clamond(Re, eD=0.0)
        return turbulent_Nunner(Re=Re, Pr=Pr, fd=fd, fd_smooth=fd_smooth)
    else:
        raise ValueError("Correlation name not recognized; see the "
                        "documentation for the available options.")


## Comparison
#import matplotlib.pyplot as plt
#import numpy as np
#from fluids.friction import friction_factor
#Pr = 0.3
#Di = 0.0254*4
#roughness = .00015
#
#methods = Nu_conv_internal_methods(Re=10000, Pr=Pr, fd=1.8E-5, x=2.5, Di=0.5)
#
#plt.figure()
#Res = np.logspace(4, 6, 300)
#for way in methods:
#    Nus = []
#    for Re in Res:
#        fd = friction_factor(Re=Re, eD=roughness/Di)
#        Nus.append(Nu_conv_internal(Re=Re, Pr=Pr, fd=fd, x=2.5, Di=0.5, Method=way))
#    plt.plot(Res, Nus, label=way)
#plt.xlabel(r'Res')
#plt.ylabel('Nus')
#plt.legend()
#
#plt.show()


### Spiral heat exchangers

def Morimoto_Hotta(Re, Pr, Dh, Rm):
    r'''Calculates Nusselt number for flow inside a spiral heat exchanger of
    spiral mean diameter `Rm` and hydraulic diameter `Dh` according to [1]_,
    also as shown in [2]_ and [3]_.

    .. math::
        Nu = 0.0239\left(1 + 5.54\frac{D_h}{R_m}\right)Re^{0.806}Pr^{0.268}

    .. math::
        D_h = \frac{2HS}{H+S}

    .. math::
        R_m = \frac{R_{min} + R_{max}}{2}


    Parameters
    ----------
    Re : float
        Reynolds number with bulk properties, [-]
    Pr : float
        Prandtl number with bulk properties [-]
    Dh : float
        Average hydraulic diameter, [m]
    Rm : float
        Average spiral radius, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Dh`, [-]

    Notes
    -----
    [1]_ is in Japanese.

    Examples
    --------
    >>> Morimoto_Hotta(1E5, 5.7, .05, .5)
    634.4879473869859

    References
    ----------
    .. [1] Morimoto, Eiji, and Kazuyuki Hotta. "Study of Geometric Structure
       and Heat Transfer Characteristics of Spiral Plate Heat Exchanger."
       Transactions of the Japan Society of Mechanical Engineers Series B 52,
       no. 474 (1986): 926-33. doi:10.1299/kikaib.52.926.
    .. [2] Bidabadi, M. and Sadaghiani, A. and Azad, A. "Spiral heat exchanger
       optimization using genetic algorithm." Transaction on Mechanical
       Engineering, International Journal of Science and Technology,
       vol. 20, no. 5 (2013): 1445-1454.
       http://www.scientiairanica.com/en/ManuscriptDetail?mid=47.
    .. [3] Turgut, Oğuz Emrah, and Mustafa Turhan Çoban. "Thermal Design of
       Spiral Heat Exchangers and Heat Pipes through Global Best Algorithm."
       Heat and Mass Transfer, July 7, 2016, 1-18.
       doi:10.1007/s00231-016-1861-y.
    '''
    return 0.0239*(1. + 5.54*Dh/Rm)*Re**0.806*Pr**0.268



### Helical/curved coils


def helical_turbulent_Nu_Mori_Nakayama(Re, Pr, Di, Dc):
    r'''Calculates Nusselt number for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Mori and Nakayama [1]_, also shown in [2]_ and [3]_.

    For :math:`Pr < 1`:

    .. math::
        Nu = \frac{Pr}{26.2(Pr^{2/3}-0.074)}Re^{0.8}\left(\frac{D_i}{D_c}
        \right)^{0.1}\left[1 + \frac{0.098}{\left[Re\left(\frac{D_i}{D_c}
        \right)^2\right]^{0.2}}\right]

    For :math:`Pr \ge 1`:

    .. math::
        Nu = \frac{Pr^{0.4}}{41}Re^{5/6}\left(\frac{D_i}{D_c}\right)^{1/12}
        \left[1 + \frac{0.061}{\left[Re\left(\frac{D_i}{D_c}\right)^{2.5}
        \right]^{1/6}}\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Pr : float
        Prandtl number with bulk properties [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Di`, [-]

    Notes
    -----
    At very low curvatures, the predicted heat transfer coefficient
    grows unbounded.

    Applicable for :math:`Re\left(\frac{D_i}{D_c}\right)^2 > 0.1`

    Examples
    --------
    >>> helical_turbulent_Nu_Mori_Nakayama(2E5, 0.7, 0.01, .2)
    496.2522480663327

    References
    ----------
    .. [1] Mori, Yasuo, and Wataru Nakayama. "Study on Forced Convective Heat
       Transfer in Curved Pipes." International Journal of Heat and Mass
       Transfer 10, no. 5 (May 1, 1967): 681-95.
       doi:10.1016/0017-9310(67)90113-5.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Hardik, B. K., P. K. Baburajan, and S. V. Prabhu. "Local Heat
       Transfer Coefficient in Helical Coils with Single Phase Flow."
       International Journal of Heat and Mass Transfer 89 (October 2015):
       522-38. doi:10.1016/j.ijheatmasstransfer.2015.05.069.
    '''
    D_ratio = Di/Dc
    if Pr < 1:
        term1 = Pr/(26.2*(Pr**(2/3.) - 0.074))*Re**0.8*D_ratio**0.1
        term2 = 1. + 0.098*(Re*D_ratio*D_ratio)**-0.2
    else:
        term1 = Pr**0.4/41.*Re**(5/6.)*(Di/Dc)**(1/12.)
        term2 = 1. + 0.061/(Re*(Di/Dc)**2.5)**(1/6.)
    return term1*term2


def helical_turbulent_Nu_Schmidt(Re, Pr, Di, Dc):
    r'''Calculates Nusselt number for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Schmidt [1]_, also shown in [2]_, [3]_, and [4]_.

    For :math:`Re_{crit} < Re < 2.2\times 10 ^4`:

    .. math::
        Nu = 0.023\left[1 + 14.8\left(1 + \frac{D_i}{D_c}\right)\left(
        \frac{D_i}{D_c}\right)^{1/3}\right]Re^{0.8-0.22\left(\frac{D_i}{D_c}
        \right)^{0.1}}Pr^{1/3}

    For :math:`2.2\times 10^4 < Re < 1.5\times 10^5`:

    .. math::
        Nu = 0.023\left[1 + 3.6\left(1 - \frac{D_i}{D_c}\right)\left(\frac{D_i}
        {D_c}\right)^{0.8}\right]Re^{0.8}Pr^{1/3}

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Pr : float
        Prandtl number with bulk properties [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Di`, [-]

    Notes
    -----
    For very low curvatures, reasonable results are returned by both cases
    of Reynolds numbers.

    Examples
    --------
    >>> helical_turbulent_Nu_Schmidt(2E5, 0.7, 0.01, .2)
    466.2569996832083

    References
    ----------
    .. [1] Schmidt, Eckehard F. "Wärmeübergang Und Druckverlust in
       Rohrschlangen." Chemie Ingenieur Technik 39, no. 13 (July 10, 1967):
       781-89. doi:10.1002/cite.330391302.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Hardik, B. K., P. K. Baburajan, and S. V. Prabhu. "Local Heat
       Transfer Coefficient in Helical Coils with Single Phase Flow."
       International Journal of Heat and Mass Transfer 89 (October 2015):
       522-38. doi:10.1016/j.ijheatmasstransfer.2015.05.069.
    .. [4] Rohsenow, Warren and James Hartnett and Young Cho. Handbook of Heat
       Transfer, 3E. New York: McGraw-Hill, 1998.
    '''
    D_ratio = Di/Dc
    if Re <= 2.2E4:
        term = Re**(0.8 - 0.22*D_ratio**0.1)*Pr**(1/3.)
        return 0.023*(1. + 14.8*(1. + D_ratio)*D_ratio**(1/3.))*term
    else:
        return 0.023*(1. + 3.6*(1. - D_ratio)*D_ratio**0.8)*Re**0.8*Pr**(1/3.)


def helical_turbulent_Nu_Xin_Ebadian(Re, Pr, Di, Dc):
    r'''Calculates Nusselt number for a fluid flowing inside a curved
    pipe such as a helical coil under turbulent conditions, using the method of
    Xin and Ebadian [1]_, also shown in [2]_ and [3]_.

    For :math:`Re_{crit} < Re < 1\times 10^5`:

    .. math::
        Nu = 0.00619Re^{0.92} Pr^{0.4}\left[1 + 3.455\left(\frac{D_i}{D_c}
        \right)\right]

    Parameters
    ----------
    Re : float
        Reynolds number with `D=Di`, [-]
    Pr : float
        Prandtl number with bulk properties [-]
    Di : float
        Inner diameter of the coil, [m]
    Dc : float
        Diameter of the helix/coil measured from the center of the tube on one
        side to the center of the tube on the other side, [m]

    Returns
    -------
    Nu : float
        Nusselt number with respect to `Di`, [-]

    Notes
    -----
    For very low curvatures, reasonable results are returned.

    The correlation was developed with data in the range of
    :math:`0.7 < Pr < 5; 0.0267 < \frac{D_i}{D_c} < 0.0884`.

    Examples
    --------
    >>> helical_turbulent_Nu_Xin_Ebadian(2E5, 0.7, 0.01, .2)
    474.11413424344755

    References
    ----------
    .. [1] Xin, R. C., and M. A. Ebadian. "The Effects of Prandtl Numbers on
       Local and Average Convective Heat Transfer Characteristics in Helical
       Pipes." Journal of Heat Transfer 119, no. 3 (August 1, 1997): 467-73.
       doi:10.1115/1.2824120.
    .. [2] El-Genk, Mohamed S., and Timothy M. Schriener. "A Review and
       Correlations for Convection Heat Transfer and Pressure Losses in
       Toroidal and Helically Coiled Tubes." Heat Transfer Engineering 0, no. 0
       (June 7, 2016): 1-28. doi:10.1080/01457632.2016.1194693.
    .. [3] Hardik, B. K., P. K. Baburajan, and S. V. Prabhu. "Local Heat
       Transfer Coefficient in Helical Coils with Single Phase Flow."
       International Journal of Heat and Mass Transfer 89 (October 2015):
       522-38. doi:10.1016/j.ijheatmasstransfer.2015.05.069.
    '''
    return 0.00619*Re**0.92*Pr**0.4*(1. + 3.455*Di/Dc)


### Rectangular Channels

def Nu_laminar_rectangular_Shan_London(a_r):
    r'''Calculates internal convection Nusselt number for laminar flows
    in a rectangular pipe of varying aspect ratio, as developed in [1]_.

    This model is derived assuming a constant wall heat flux from all sides.
    This is entirely theoretically derived and reproduced experimentally.

    .. math::
        Nu_{lam} = 8.235\left(1 - 2.0421\alpha + 3.0853\alpha^2
        - 2.4765\alpha^3 + 1.0578\alpha^4 - 0.1861\alpha^5\right)

    Parameters
    ----------
    a_r : float
        The aspect ratio of the channel, from 0 to 1 [-]

    Returns
    -------
    Nu : float
        Nusselt number of flow in a rectangular channel, [-]

    Notes
    -----
    At an aspect ratio of 1 (square channel), the Nusselt number converges to
    3.610224. The authors of [1]_ also published [2]_, which tabulates in
    their table 42 some less precise results that are used to check this
    function.

    Examples
    --------
    >>> Nu_laminar_rectangular_Shan_London(.7)
    3.751762675455

    References
    ----------
    .. [1] Shah, R. K, and Alexander Louis London. Supplement 1: Laminar Flow
       Forced Convection in Ducts: A Source Book for Compact Heat Exchanger
       Analytical Data. New York: Academic Press, 1978.
    .. [2] Shah, Ramesh K., and A. L. London. "Laminar Flow Forced Convection
       Heat Transfer and Flow Friction in Straight and Curved Ducts - A Summary
       of Analytical Solutions." STANFORD UNIV CA DEPT OF MECHANICAL
       ENGINEERING, STANFORD UNIV CA DEPT OF MECHANICAL ENGINEERING, November
       1971. http://www.dtic.mil/docs/citations/AD0736260.
    '''
    return 8.235*(1 - 2.0421*a_r + 3.0853*a_r**2 - 2.4765*a_r**3
                  + 1.0578*a_r**4 - 0.1861*a_r**5)

