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

from math import atan, exp, log10, pi

from fluids.constants import g
from fluids.core import Boiling, Bond, Prandtl, Weber
from fluids.numerics import secant
from fluids.two_phase_voidage import Lockhart_Martinelli_Xtt

from ht.boiling_nucleic import Cooper, Forster_Zuber
from ht.conv_internal import turbulent_Dittus_Boelter, turbulent_Gnielinski

__all__ = ['Thome', 'Liu_Winterton', 'Chen_Edelstein', 'Chen_Bennett',
           'Lazarek_Black', 'Li_Wu', 'Sun_Mishima', 'Yun_Heo_Kim']

__numba_additional_funcs__ = ('to_solve_q_Thome',)

def Lazarek_Black(m, D, mul, kl, Hvap, q=None, Te=None):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in vertical tubes for either upward or downward flow. Correlation
    is as shown in [1]_, and also reviewed in [2]_ and [3]_.

    Either the heat flux or excess temperature is required for the calculation
    of heat transfer coefficient.

    Quality independent. Requires no properties of the gas.
    Uses a Reynolds number assuming all the flow is liquid.

    .. math::
        h_{tp} = 30 Re_{lo}^{0.857} Bg^{0.714} \frac{k_l}{D}

    .. math::
        Re_{lo} = \frac{G_{tp}D}{\mu_l}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    D : float
        Diameter of the channel [m]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    q : float, optional
        Heat flux to wall [W/m^2]
    Te : float, optional
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ has been reviewed.

    [2]_ claims it was developed for a range of quality 0-0.6,
    Relo 860-5500, mass flux 125-750 kg/m^2/s, q of 1.4-38 W/cm^2, and with a
    pipe diameter of 3.1 mm. Developed with data for R113 only.

    Examples
    --------
    >>> Lazarek_Black(m=10, D=0.3, mul=1E-3, kl=0.6, Hvap=2E6, Te=100)
    9501.932636079293

    References
    ----------
    .. [1] Lazarek, G. M., and S. H. Black. "Evaporative Heat Transfer,
       Pressure Drop and Critical Heat Flux in a Small Vertical Tube with
       R-113." International Journal of Heat and Mass Transfer 25, no. 7 (July
       1982): 945-60. doi:10.1016/0017-9310(82)90070-9.
    .. [2] Fang, Xiande, Zhanru Zhou, and Dingkun Li. "Review of Correlations
       of Flow Boiling Heat Transfer Coefficients for Carbon Dioxide."
       International Journal of Refrigeration 36, no. 8 (December 2013):
       2017-39. doi:10.1016/j.ijrefrig.2013.05.015.
    .. [3] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella.
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    G = m/(pi/4*D**2)
    Relo = G*D/mul
    if q is not None:
        Bg = Boiling(G=G, q=q, Hvap=Hvap)
        return 30*Relo**0.857*Bg**0.714*kl/D
    elif Te is not None:
        # Solved with sympy
        return 27000*30**(71/143)*(1./(G*Hvap))**(357/143)*Relo**(857/286)*Te**(357/143)*kl**(500/143)/D**(500/143)
    else:
        raise ValueError('Either q or Te is needed for this correlation')


def Li_Wu(m, x, D, rhol, rhog, mul, kl, Hvap, sigma, q=None, Te=None):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is as shown in [1]_, and also reviewed in [2]_ and [3]_.

    Either the heat flux or excess temperature is required for the calculation
    of heat transfer coefficient. Uses liquid Reynolds number, Bond number,
    and Boiling number.

    .. math::
        h_{tp} = 334 Bg^{0.3}(Bo\cdot Re_l^{0.36})^{0.4}\frac{k_l}{D}

    .. math::
        Re_{l} = \frac{G(1-x)D}{\mu_l}

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
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    q : float, optional
        Heat flux to wall [W/m^2]
    Te : float, optional
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ has been reviewed.

    [1]_ used 18 sets of experimental data to derive the results, covering
    hydraulic diameters from 0.19 to 3.1 mm and 12 different fluids.

    Examples
    --------
    >>> Li_Wu(m=1, x=0.2, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, q=1E5)
    5345.409399239492

    References
    ----------
    .. [1] Li, Wei, and Zan Wu. "A General Correlation for Evaporative Heat
       Transfer in Micro/mini-Channels." International Journal of Heat and Mass
       Transfer 53, no. 9-10 (April 2010): 1778-87.
       doi:10.1016/j.ijheatmasstransfer.2010.01.012.
    .. [2] Fang, Xiande, Zhanru Zhou, and Dingkun Li. "Review of Correlations
       of Flow Boiling Heat Transfer Coefficients for Carbon Dioxide."
       International Journal of Refrigeration 36, no. 8 (December 2013):
       2017-39. doi:10.1016/j.ijrefrig.2013.05.015.
    .. [3] Kim, Sung-Min, and Issam Mudawar. "Review of Databases and
       Predictive Methods for Pressure Drop in Adiabatic, Condensing and
       Boiling Mini/micro-Channel Flows." International Journal of Heat and
       Mass Transfer 77 (October 2014): 74-97.
       doi:10.1016/j.ijheatmasstransfer.2014.04.035.
    '''
    G = m/(pi/4*D**2)
    Rel = G*D*(1-x)/mul
    Bo = Bond(rhol=rhol, rhog=rhog, sigma=sigma, L=D)
    if q is not None:
        Bg = Boiling(G=G, q=q, Hvap=Hvap)
        return 334*Bg**0.3*(Bo*Rel**0.36)**0.4*kl/D
    elif Te is not None:
        A = 334*(Bo*Rel**0.36)**0.4*kl/D
        return A**(10/7.)*Te**(3/7.)/(G**(3/7.)*Hvap**(3/7.))
    else:
        raise ValueError('Either q or Te is needed for this correlation')


def Sun_Mishima(m, D, rhol, rhog, mul, kl, Hvap, sigma, q=None, Te=None):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is as shown in [1]_, and also reviewed in [2]_.

    Either the heat flux or excess temperature is required for the calculation
    of heat transfer coefficient. Uses liquid-only Reynolds number, Weber
    number, and Boiling number. Weber number is defined in terms of the velocity
    if all fluid were liquid.

    .. math::
        h_{tp} = \frac{ 6 Re_{lo}^{1.05} Bg^{0.54}}
        {We_l^{0.191}(\rho_l/\rho_g)^{0.142}}\frac{k_l}{D}

    .. math::
        Re_{lo} = \frac{G_{tp}D}{\mu_l}

    Parameters
    ----------
    m : float
        Mass flow rate [kg/s]
    D : float
        Diameter of the tube [m]
    rhol : float
        Density of the liquid [kg/m^3]
    rhog : float
        Density of the gas [kg/m^3]
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    q : float, optional
        Heat flux to wall [W/m^2]
    Te : float, optional
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ has been reviewed.

    [1]_ used 2501 data points to derive the results, covering
    hydraulic diameters from 0.21 to 6.05 mm and 11 different fluids.


    Examples
    --------
    >>> Sun_Mishima(m=1, D=0.3, rhol=567., rhog=18.09, kl=0.086, mul=156E-6, sigma=0.02, Hvap=9E5, Te=10)
    507.6709168372167

    References
    ----------
    .. [1] Sun, Licheng, and Kaichiro Mishima. "An Evaluation of Prediction
       Methods for Saturated Flow Boiling Heat Transfer in Mini-Channels."
       International Journal of Heat and Mass Transfer 52, no. 23-24 (November
       2009): 5323-29. doi:10.1016/j.ijheatmasstransfer.2009.06.041.
    .. [2] Fang, Xiande, Zhanru Zhou, and Dingkun Li. "Review of Correlations
       of Flow Boiling Heat Transfer Coefficients for Carbon Dioxide."
       International Journal of Refrigeration 36, no. 8 (December 2013):
       2017-39. doi:10.1016/j.ijrefrig.2013.05.015.
    '''
    G = m/(pi/4*D**2)
    V = G/rhol
    Relo = G*D/mul
    We = Weber(V=V, L=D, rho=rhol, sigma=sigma)
    if q is not None:
        Bg = Boiling(G=G, q=q, Hvap=Hvap)
        return 6*Relo**1.05*Bg**0.54/(We**0.191*(rhol/rhog)**0.142)*kl/D
    elif Te is not None:
        A = 6*Relo**1.05/(We**0.191*(rhol/rhog)**0.142)*kl/D
        return A**(50/23.)*Te**(27/23.)/(G**(27/23.)*Hvap**(27/23.))
    else:
        raise ValueError('Either q or Te is needed for this correlation')


def Thome(m, x, D, rhol, rhog, mul, mug, kl, kg, Cpl, Cpg, Hvap, sigma, Psat,
          Pc, q=None, Te=None):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is as developed in [1]_ and [2]_, and also reviewed [3]_. This is a
    complicated model, but expected to have more accuracy as a result.

    Either the heat flux or excess temperature is required for the calculation
    of heat transfer coefficient. The solution for a specified excess
    temperature is solved numerically, making it slow.

    .. math::
        h(z) = \frac{t_l}{\tau} h_l(z) +\frac{t_{film}}{\tau} h_{film}(z)
        +  \frac{t_{dry}}{\tau} h_{g}(z)

    .. math::
        h_{l/g}(z) = (Nu_{lam}^4 + Nu_{trans}^4)^{1/4} k/D

    .. math::
        Nu_{laminar} = 0.91 {Pr}^{1/3} \sqrt{ReD/L(z)}

    .. math::
        Nu_{trans} = \frac{ (f/8) (Re-1000)Pr}{1+12.7 (f/8)^{1/2} (Pr^{2/3}-1)}
        \left[ 1 + \left( \frac{D}{L(z)}\right)^{2/3}\right]

    .. math::
        f = (1.82 \log_{10} Re - 1.64 )^{-2}

    .. math::
        L_l = \frac{\tau G_{tp}}{\rho_l}(1-x)

    .. math::
        L_{dry} = v_p t_{dry}

    .. math::
        t_l = \frac{\tau}{1 + \frac{\rho_l}{\rho_g}\frac{x}{1-x}}

    .. math::
        t_v = \frac{\tau}{1 + \frac{\rho_g}{\rho_l}\frac{1-x}{x}}

    .. math::
        \tau = \frac{1}{f_{opt}}

    .. math::
        f_{opt} = \left(\frac{q}{q_{ref}}\right)^{n_f}

    .. math::
        q_{ref} = 3328\left(\frac{P_{sat}}{P_c}\right)^{-0.5}

    .. math::
        t_{dry,film} = \frac{\rho_l \Delta H_{vap}}{q}[\delta_0(z) -
        \delta_{min}]

    .. math::
        \frac{\delta_0}{D} = C_{\delta 0}\left(3\sqrt{\frac{\nu_l}{v_p D}}
        \right)^{0.84}\left[(0.07Bo^{0.41})^{-8} + 0.1^{-8}\right]^{-1/8}

    .. math::
        Bo = \frac{\rho_l D}{\sigma} v_p^2

    .. math::
        v_p = G_{tp} \left[\frac{x}{\rho_g} + \frac{1-x}{\rho_l}\right]

    .. math::
        h_{film}(z) = \frac{2 k_l}{\delta_0(z) + \delta_{min}(z)}

    .. math::
        \delta_{min} = 0.3\cdot 10^{-6} \text{m}

    .. math::
        C_{\delta,0} = 0.29

    .. math::
        n_f = 1.74

    if t dry film > tv:

    .. math::
        \delta_{end}(x) = \delta(z, t_v)

    .. math::
        t_{film} = t_v

    .. math::
        t_{dry} = 0

    Otherwise:

    .. math::
        \delta_{end}(z) = \delta_{min}

    .. math::
        t_{film} = t_{dry,film}

    .. math::
        t_{dry} = t_v - t_{film}

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
    mul : float
        Viscosity of liquid [Pa*s]
    mug : float
        Viscosity of gas [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    kg : float
        Thermal conductivity of gas [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Cpg : float
        Heat capacity of gas [J/kg/K]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    Psat : float
        Vapor pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    q : float, optional
        Heat flux to wall [W/m^2]
    Te : float, optional
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ and [2]_ have been reviewed, and are accurately reproduced in [3]_.

    [1]_ used data from 7 studies, covering 7 fluids and Dh from 0.7-3.1 mm,
    heat flux from 0.5-17.8 W/cm^2, x from 0.01-0.99, and G from 50-564
    kg/m^2/s.

    Liquid and/or gas slugs are both considered, and are hydrodynamically
    developing. `Ll` is the calculated length of liquid slugs, and `L_dry`
    is the same for vapor slugs.

    Because of the complexity of the model and that there is some logic in this
    function, `Te` as an input may lead to a different solution that the
    calculated `q` will in return.

    Examples
    --------
    >>> Thome(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086, kg=0.2,
    ... mul=156E-6, mug=1E-5, Cpl=2300, Cpg=1400, sigma=0.02, Hvap=9E5,
    ... Psat=1E5, Pc=22E6, q=1E5)
    1633.008836502032

    References
    ----------
    .. [1] Thome, J. R., V. Dupont, and A. M. Jacobi. "Heat Transfer Model for
       Evaporation in Microchannels. Part I: Presentation of the Model."
       International Journal of Heat and Mass Transfer 47, no. 14-16 (July
       2004): 3375-85. doi:10.1016/j.ijheatmasstransfer.2004.01.006.
    .. [2] Dupont, V., J. R. Thome, and A. M. Jacobi. "Heat Transfer Model for
       Evaporation in Microchannels. Part II: Comparison with the Database."
       International Journal of Heat and Mass Transfer 47, no. 14-16 (July
       2004): 3387-3401. doi:10.1016/j.ijheatmasstransfer.2004.01.007.
    .. [3] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella.
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    if q is None and Te is not None:
        q = secant(to_solve_q_Thome, 1E4, args=( m, x, D, rhol, rhog, kl, kg, mul, mug, Cpl, Cpg, sigma, Hvap, Psat, Pc, Te))
        return Thome(m=m, x=x, D=D, rhol=rhol, rhog=rhog, kl=kl, kg=kg, mul=mul, mug=mug, Cpl=Cpl, Cpg=Cpg, sigma=sigma, Hvap=Hvap, Psat=Psat, Pc=Pc, q=q)
    elif q is None and Te is None:
        raise ValueError('Either q or Te is needed for this correlation')
    C_delta0 = 0.3E-6
    G = m/(pi/4*D**2)
    Rel = G*D*(1-x)/mul
    Reg = G*D*x/mug
    qref = 3328*(Psat/Pc)**-0.5
    if q is None:
        q = 1e4 # Make numba happy, their bug, never gets ran
    fopt = (q/qref)**1.74
    tau = 1./fopt

    vp = G*(x/rhog + (1-x)/rhol)
    Bo = rhol*D/sigma*vp**2 # Not standard definition
    nul = mul/rhol
    delta0 = D*0.29*(3*(nul/vp/D)**0.5)**0.84*((0.07*Bo**0.41)**-8 + 0.1**-8)**(-1/8.)

    tl = tau/(1 + rhol/rhog*(x/(1.-x)))
    tv = tau/(1 + rhog/rhol*((1.-x)/x))

    t_dry_film = rhol*Hvap/q*(delta0 - C_delta0)
    if t_dry_film > tv:
        t_film = tv
        delta_end = delta0 - q/rhol/Hvap*tv # what could time possibly be?
        t_dry = 0
    else:
        t_film = t_dry_film
        delta_end = C_delta0
        t_dry = tv-t_film
    Ll = tau*G/rhol*(1-x)
    Ldry = t_dry*vp


    Prg = Prandtl(Cp=Cpg, k=kg, mu=mug)
    Prl = Prandtl(Cp=Cpl, k=kl, mu=mul)
    fg = (1.82*log10(Reg) - 1.64)**-2
    fl = (1.82*log10(Rel) - 1.64)**-2

    Nu_lam_Zl = 2*0.455*(Prl)**(1/3.)*(D*Rel/Ll)**0.5
    Nu_trans_Zl = turbulent_Gnielinski(Re=Rel, Pr=Prl, fd=fl)*(1 + (D/Ll)**(2/3.))
    if Ldry == 0:
        Nu_lam_Zg, Nu_trans_Zg = 0, 0
    else:
        Nu_lam_Zg = 2*0.455*(Prg)**(1/3.)*(D*Reg/Ldry)**0.5
        Nu_trans_Zg = turbulent_Gnielinski(Re=Reg, Pr=Prg, fd=fg)*(1 + (D/Ldry)**(2/3.))

    h_Zg = kg/D*(Nu_lam_Zg**4 + Nu_trans_Zg**4)**0.25
    h_Zl = kl/D*(Nu_lam_Zl**4 + Nu_trans_Zl**4)**0.25

    h_film = 2*kl/(delta0 + C_delta0)
    return tl/tau*h_Zl + t_film/tau*h_film + t_dry/tau*h_Zg

def to_solve_q_Thome(q, m, x, D, rhol, rhog, kl, kg, mul, mug, Cpl, Cpg, sigma, Hvap, Psat, Pc, Te):
    err = q/Thome(m=m, x=x, D=D, rhol=rhol, rhog=rhog, kl=kl, kg=kg, mul=mul, mug=mug, Cpl=Cpl, Cpg=Cpg, sigma=sigma, Hvap=Hvap, Psat=Psat, Pc=Pc, q=q) - Te
    return err

def Yun_Heo_Kim(m, x, D, rhol, mul, Hvap, sigma, q=None, Te=None):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is as shown in [1]_ and [2]_, and also reviewed in [3]_.

    Either the heat flux or excess temperature is required for the calculation
    of heat transfer coefficient. Uses liquid Reynolds number, Weber
    number, and Boiling number. Weber number is defined in terms of the velocity
    if all fluid were liquid.

    .. math::
        h_{tp} = 136876(Bg\cdot We_l)^{0.1993} Re_l^{-0.1626}

    .. math::
        Re_l = \frac{G D (1-x)}{\mu_l}

    .. math::
        We_l = \frac{G^2 D}{\rho_l \sigma}

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
    mul : float
        Viscosity of liquid [Pa*s]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    q : float, optional
        Heat flux to wall [W/m^2]
    Te : float, optional
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ has been reviewed.

    Examples
    --------
    >>> Yun_Heo_Kim(m=1, x=0.4, D=0.3, rhol=567., mul=156E-6, sigma=0.02, Hvap=9E5, q=1E4)
    9479.313988550184

    References
    ----------
    .. [1] Yun, Rin, Jae Hyeok Heo, and Yongchan Kim. "Evaporative Heat
       Transfer and Pressure Drop of R410A in Microchannels." International
       Journal of Refrigeration 29, no. 1 (January 2006): 92-100.
       doi:10.1016/j.ijrefrig.2005.08.005.
    .. [2] Yun, Rin, Jae Hyeok Heo, and Yongchan Kim. "Erratum to 'Evaporative
       Heat Transfer and Pressure Drop of R410A in Microchannels; [Int. J.
       Refrigeration 29 (2006) 92-100]." International Journal of Refrigeration
       30, no. 8 (December 2007): 1468. doi:10.1016/j.ijrefrig.2007.08.003.
    .. [3] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella.
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    G = m/(pi/4*D**2)
    V = G/rhol
    Rel = G*D*(1-x)/mul
    We = Weber(V=V, L=D, rho=rhol, sigma=sigma)
    if q is not None:
        Bg = Boiling(G=G, q=q, Hvap=Hvap)
        return 136876*(Bg*We)**0.1993*Rel**-0.1626
    elif Te is not None:
        A = 136876*(We)**0.1993*Rel**-0.1626*(Te/G/Hvap)**0.1993
        return A**(10000/8007.)
    else:
        raise ValueError('Either q or Te is needed for this correlation')


def Chen_Edelstein(m, x, D, rhol, rhog, mul, mug, kl, Cpl, Hvap, sigma,
                   dPsat, Te):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is developed in [1]_ and [2]_, and reviewed in [3]_. This model is one of
    the most often used. It uses the Dittus-Boelter correlation for turbulent
    convection and the Forster-Zuber correlation for pool boiling, and
    combines them with two factors `F` and `S`.


    .. math::
        h_{tp} = S\cdot h_{nb} + F \cdot h_{sp,l}

    .. math::
        h_{sp,l} = 0.023 Re_l^{0.8} Pr_l^{0.4} k_l/D

    .. math::
        Re_l = \frac{DG(1-x)}{\mu_l}

    .. math::
        h_{nb} = 0.00122\left( \frac{\lambda_l^{0.79} c_{p,l}^{0.45}
        \rho_l^{0.49}}{\sigma^{0.5} \mu^{0.29} H_{vap}^{0.24} \rho_g^{0.24}}
        \right)\Delta T_{sat}^{0.24} \Delta p_{sat}^{0.75}

    .. math::
        F = (1 + X_{tt}^{-0.5})^{1.78}

    .. math::
        X_{tt} = \left( \frac{1-x}{x}\right)^{0.9} \left(\frac{\rho_g}{\rho_l}
        \right)^{0.5}\left( \frac{\mu_l}{\mu_g}\right)^{0.1}

    .. math::
        S = 0.9622 - 0.5822\left(\tan^{-1}\left(\frac{Re_L\cdot F^{1.25}}
        {6.18\cdot 10^4}\right)\right)

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
    mul : float
        Viscosity of liquid [Pa*s]
    mug : float
        Viscosity of gas [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    dPsat : float
        Difference in Saturation pressure of fluid at Te and T, [Pa]
    Te : float
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ and [2]_ have been reviewed, but the model is only put together in
    the review of [3]_. Many other forms of this equation exist with different
    functions for `F` and `S`.

    Examples
    --------
    >>> Chen_Edelstein(m=0.106, x=0.2, D=0.0212, rhol=567, rhog=18.09,
    ... mul=156E-6, mug=7.11E-6, kl=0.086, Cpl=2730, Hvap=2E5, sigma=0.02,
    ... dPsat=1E5, Te=3)
    3289.058731974052

    See Also
    --------
    turbulent_Dittus_Boelter
    Forster_Zuber

    References
    ----------
    .. [1] Chen, J. C. "Correlation for Boiling Heat Transfer to Saturated
       Fluids in Convective Flow." Industrial & Engineering Chemistry Process
       Design and Development 5, no. 3 (July 1, 1966): 322-29.
       doi:10.1021/i260019a023.
    .. [2] Edelstein, Sergio, A. J. Pérez, and J. C. Chen. "Analytic
       Representation of Convective Boiling Functions." AIChE Journal 30, no.
       5 (September 1, 1984): 840-41. doi:10.1002/aic.690300528.
    .. [3] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella.
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    G = m/(pi/4*D**2)
    Rel = D*G*(1-x)/mul
    Prl = Prandtl(Cp=Cpl, mu=mul, k=kl)
    hl = turbulent_Dittus_Boelter(Re=Rel, Pr=Prl)*kl/D

    Xtt = Lockhart_Martinelli_Xtt(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug)
    F = (1 + Xtt**-0.5)**1.78
    Re = Rel*F**1.25
    S = 0.9622 - 0.5822*atan(Re/6.18E4)
    hnb = Forster_Zuber(Te=Te, dPsat=dPsat, Cpl=Cpl, kl=kl, mul=mul, sigma=sigma,
                       Hvap=Hvap, rhol=rhol, rhog=rhog)
    return hnb*S + hl*F


def Chen_Bennett(m, x, D, rhol, rhog, mul, mug, kl, Cpl, Hvap, sigma,
                   dPsat, Te):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is developed in [1]_ and [2]_, and reviewed in [3]_. This model is one of
    the most often used, and replaces the `Chen_Edelstein` correlation. It uses
    the Dittus-Boelter correlation for turbulent convection and the
    Forster-Zuber correlation for pool boiling, and combines them with two
    factors `F` and `S`.

    .. math::
        h_{tp} = S\cdot h_{nb} + F \cdot h_{sp,l}

    .. math::
       h_{sp,l} = 0.023 Re_l^{0.8} Pr_l^{0.4} k_l/D

    .. math::
       Re_l = \frac{DG(1-x)}{\mu_l}

    .. math::
       h_{nb} = 0.00122\left( \frac{\lambda_l^{0.79} c_{p,l}^{0.45}
        \rho_l^{0.49}}{\sigma^{0.5} \mu^{0.29} H_{vap}^{0.24} \rho_g^{0.24}}
        \right)\Delta T_{sat}^{0.24} \Delta p_{sat}^{0.75}

    .. math::
       F = \left(\frac{Pr_1+1}{2}\right)^{0.444}\cdot (1+X_{tt}^{-0.5})^{1.78}

    .. math::
       S = \frac{1-\exp(-F\cdot h_{conv} \cdot X_0/k_l)}
        {F\cdot h_{conv}\cdot X_0/k_l}

    .. math::
       X_{tt} = \left( \frac{1-x}{x}\right)^{0.9} \left(\frac{\rho_g}{\rho_l}
        \right)^{0.5}\left( \frac{\mu_l}{\mu_g}\right)^{0.1}

    .. math::
       X_0 = 0.041 \left(\frac{\sigma}{g \cdot (\rho_l-\rho_v)}\right)^{0.5}

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
    mul : float
        Viscosity of liquid [Pa*s]
    mug : float
        Viscosity of gas [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    Hvap : float
        Heat of vaporization of liquid [J/kg]
    sigma : float
        Surface tension of liquid [N/m]
    dPsat : float
        Difference in Saturation pressure of fluid at Te and T, [Pa]
    Te : float
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ and [2]_ have been reviewed, but the model is only put together in
    the review of [3]_. Many other forms of this equation exist with different
    functions for `F` and `S`.

    Examples
    --------
    >>> Chen_Bennett(m=0.106, x=0.2, D=0.0212, rhol=567, rhog=18.09,
    ... mul=156E-6, mug=7.11E-6, kl=0.086, Cpl=2730, Hvap=2E5, sigma=0.02,
    ... dPsat=1E5, Te=3)
    4938.275351219369

    See Also
    --------
    Chen_Edelstein
    turbulent_Dittus_Boelter
    Forster_Zuber

    References
    ----------
    .. [1] Bennett, Douglas L., and John C. Chen. "Forced Convective Boiling in
       Vertical Tubes for Saturated Pure Components and Binary Mixtures."
       AIChE Journal 26, no. 3 (May 1, 1980): 454-61. doi:10.1002/aic.690260317.
    .. [2] Bennett, Douglas L., M.W. Davies and B.L. Hertzler, The Suppression
       of Saturated Nucleate Boiling by Forced Convective Flow, American
       Institute of Chemical Engineers Symposium Series, vol. 76, no. 199.
       91-103, 1980.
    .. [3] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella.
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    G = m/(pi/4*D**2)
    Rel = D*G*(1-x)/mul
    Prl = Prandtl(Cp=Cpl, mu=mul, k=kl)
    hl = turbulent_Dittus_Boelter(Re=Rel, Pr=Prl)*kl/D
    Xtt = Lockhart_Martinelli_Xtt(x=x, rhol=rhol, rhog=rhog, mul=mul, mug=mug)
    F = ((Prl+1)/2.)**0.444*(1 + Xtt**-0.5)**1.78
    X0 = 0.041*(sigma/(g*(rhol-rhog)))**0.5
    S = (1 - exp(-F*hl*X0/kl))/(F*hl*X0/kl)

    hnb = Forster_Zuber(Te=Te, dPsat=dPsat, Cpl=Cpl, kl=kl, mul=mul, sigma=sigma,
                       Hvap=Hvap, rhol=rhol, rhog=rhog)
    return hnb*S + hl*F


def Liu_Winterton(m, x, D, rhol, rhog, mul, kl, Cpl, MW, P,  Pc, Te):
    r'''Calculates heat transfer coefficient for film boiling of saturated
    fluid in any orientation of flow. Correlation
    is as developed in [1]_, also reviewed in [2]_ and [3]_.

    Excess wall temperature is required to use this correlation.

    .. math::
        h_{tp} = \sqrt{ (F\cdot h_l)^2 + (S\cdot h_{nb})^2}

    .. math::
       S = \left( 1+0.055F^{0.1} Re_{L}^{0.16}\right)^{-1}

    .. math::
       h_{l} = 0.023 Re_L^{0.8} Pr_l^{0.4} k_l/D

    .. math::
       Re_L = \frac{GD}{\mu_l}

    .. math::
       F = \left[ 1+ xPr_{l}(\rho_l/\rho_g-1)\right]^{0.35}

    .. math::
       h_{nb} = \left(55\Delta Te^{0.67} \frac{P}{P_c}^{(0.12 - 0.2\log_{10}
         R_p)}(-\log_{10} \frac{P}{P_c})^{-0.55} MW^{-0.5}\right)^{1/0.33}

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
    mul : float
        Viscosity of liquid [Pa*s]
    kl : float
        Thermal conductivity of liquid [W/m/K]
    Cpl : float
        Heat capacity of liquid [J/kg/K]
    MW : float
        Molecular weight of the fluid, [g/mol]
    P : float
        Pressure of fluid, [Pa]
    Pc : float
        Critical pressure of fluid, [Pa]
    Te : float, optional
        Excess temperature of wall, [K]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/m^2/K]

    Notes
    -----
    [1]_ has been reviewed, and is accurately reproduced in [3]_.

    Uses the `Cooper` and `turbulent_Dittus_Boelter` correlations.

    A correction for horizontal flow at low Froude numbers is available in
    [1]_ but has not been implemented and is not recommended in several
    sources.

    Examples
    --------
    >>> Liu_Winterton(m=1, x=0.4, D=0.3, rhol=567., rhog=18.09, kl=0.086,
    ... mul=156E-6, Cpl=2300, P=1E6, Pc=22E6, MW=44.02, Te=7)
    4747.749477190532

    References
    ----------
    .. [1] Liu, Z., and R. H. S. Winterton. "A General Correlation for
       Saturated and Subcooled Flow Boiling in Tubes and Annuli, Based on a
       Nucleate Pool Boiling Equation." International Journal of Heat and Mass
       Transfer 34, no. 11 (November 1991): 2759-66.
       doi:10.1016/0017-9310(91)90234-6.
    .. [2] Fang, Xiande, Zhanru Zhou, and Dingkun Li. "Review of Correlations
       of Flow Boiling Heat Transfer Coefficients for Carbon Dioxide."
       International Journal of Refrigeration 36, no. 8 (December 2013):
       2017-39. doi:10.1016/j.ijrefrig.2013.05.015.
    .. [3] Bertsch, Stefan S., Eckhard A. Groll, and Suresh V. Garimella.
       "Review and Comparative Analysis of Studies on Saturated Flow Boiling in
       Small Channels." Nanoscale and Microscale Thermophysical Engineering 12,
       no. 3 (September 4, 2008): 187-227. doi:10.1080/15567260802317357.
    '''
    G = m/(pi/4*D**2)
    ReL = D*G/mul
    Prl = Prandtl(Cp=Cpl, mu=mul, k=kl)
    hl = turbulent_Dittus_Boelter(Re=ReL, Pr=Prl)*kl/D
    F = (1 + x*Prl*(rhol/rhog - 1))**0.35
    S = (1 + 0.055*F**0.1*ReL**0.16)**-1
    h_nb = Cooper(Te=Te, P=P, Pc=Pc, MW=MW)
    return ((F*hl)**2 + (S*h_nb)**2)**0.5
