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

from math import log, pi

from fluids.constants import g
from fluids.friction import friction_factor

__all__ = ['Lehrer', 'Stein_Schmidt']

def Lehrer(m, Dtank, Djacket, H, Dinlet, rho, Cp, k, mu, muw=None,
           isobaric_expansion=None, dT=None, inlettype='tangential',
           inletlocation='auto'):
    r'''Calculates average heat transfer coefficient for a jacket around a
    vessel according to [1]_ as described in [2]_.

    .. math::
        Nu_{S,L} = \left[\frac{0.03Re_S^{0.75}Pr}{1 + \frac{1.74(Pr-1)}
        {Re_S^{0.125}}}\right]\left(\frac{\mu}{\mu_w}\right)^{0.14}

    .. math::
        d_g = \left(\frac{8}{3}\right)^{0.5}\delta

    .. math::
        v_h = (v_Sv_{inlet})^{0.5} + v_A

    .. math::
        v_{inlet} = \frac{Q}{\frac{\pi}{4}d_{inlet}^2}

    .. math::
        v_s = \frac{Q}{\frac{\pi}{4}(D_{jacket}^2 - D_{tank}^2)}

    For Radial inlets:

    .. math::
        v_A = 0.5(2g H \beta\delta \Delta T)^{0.5}

    For Tangential inlets:

    .. math::
        v_A = 0

    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/s]
    Dtank : float
        Outer diameter of tank or vessel surrounded by jacket, [m]
    Djacket : float
        Inner diameter of jacket surrounding a vessel or tank, [m]
    H : float
        Height of the vessel or tank, [m]
    Dinlet : float
        Inner diameter of inlet into the jacket, [m]
    rho : float
        Density of the fluid at Tm [kg/m^3]
    Cp : float
        Heat capacity of fluid at Tm [J/kg/K]
    k : float
        Thermal conductivity of fluid at Tm [W/m/K]
    mu : float
        Viscosity of fluid at Tm [Pa*s]
    muw : float, optional
        Viscosity of fluid at Tw [Pa*s]
    isobaric_expansion : float, optional
        Constant pressure expansivity of a fluid, [m^3/mol/K]
    dT : float, optional
        Temperature difference of fluid in jacket, [K]
    inlettype : str, optional
        Either 'tangential' or 'radial'
    inletlocation : str, optional
        Either 'top' or 'bottom' or 'auto'

    Returns
    -------
    h : float
        Average heat transfer coefficient inside the jacket [W/m^2/K]

    Notes
    -----
    If the fluid is heated and enters from the bottom, natural convection
    assists the heat transfer and the Grashof term is added; if it were to enter
    from the top, it would be subtracted. The situation is reversed if entry
    is from the top.

    Examples
    --------
    Example as in [2]_, matches completely.

    >>> Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20.,
    ... rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6)
    2922.128124761829

    Examples similar to in [2]_ but covering the other case:

    >>> Lehrer(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025, dT=20.,
    ... rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6,
    ... inlettype='radial', isobaric_expansion=0.000303)
    3269.4389632666557

    References
    ----------
    .. [1] Lehrer, Isaac H. "Jacket-Side Nusselt Number." Industrial &
       Engineering Chemistry Process Design and Development 9, no. 4
       (October 1, 1970): 553-58. doi:10.1021/i260036a010.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    delta = (Djacket-Dtank)/2.
    Q = m/rho
    Pr = Cp*mu/k
    vs = Q/H/delta
    vo = Q/(pi/4*Dinlet**2)
    if dT is not None and isobaric_expansion is not None and inlettype == 'radial' and inletlocation is not None:
        if dT > 0: # Heating jacket fluid
            if inletlocation == 'auto' or inletlocation == 'bottom':
                va = 0.5*(2*g*H*isobaric_expansion*abs(dT))**0.5
            else:
                va = -0.5*(2*g*H*isobaric_expansion*abs(dT))**0.5
        else: # cooling fluid
            if inletlocation == 'auto' or inletlocation == 'top':
                va = 0.5*(2*g*H*isobaric_expansion*abs(dT))**0.5
            else:
                va = -0.5*(2*g*H*isobaric_expansion*abs(dT))**0.5
    else:
        va = 0
    vh = (vs*vo)**0.5 + va
    dg = (8/3.)**0.5*delta
    Res = vh*dg*rho/mu
    if muw is not None:
        NuSL = (0.03*Res**0.75*Pr)/(1 + 1.74*(Pr-1)/Res**0.125)*(mu/muw)**0.14
    else:
        NuSL = (0.03*Res**0.75*Pr)/(1 + 1.74*(Pr-1)/Res**0.125)
    return NuSL*k/dg


def Stein_Schmidt(m, Dtank, Djacket, H, Dinlet,
                  rho, Cp, k, mu, muw=None, rhow=None,
                  inlettype='tangential', inletlocation='auto', roughness=0.0):
    r'''Calculates average heat transfer coefficient for a jacket around a
    vessel according to [1]_ as described in [2]_.

    .. math::
        l_{ch} = \left[\left(\frac{\pi}{2}\right)^2 D_{tank}^2+H^2\right]^{0.5}

    .. math::
        d_{ch} = 2\delta

    .. math::
        Re_j = \frac{v_{ch}d_{ch}\rho}{\mu}

    .. math::
        Gr_J = \frac{g\rho(\rho-\rho_w)d_{ch}^3}{\mu^2}

    .. math::
        Re_{J,eq} = \left[Re_J^2\pm \left(\frac{|Gr_J|\frac{H}{d_{ch}}}{50}
        \right)\right]^{0.5}

    .. math::
        Nu_J = (Nu_A^3 + Nu_B^3 + Nu_C^3 + Nu_D^3)^{1/3}\left(\frac{\mu}
        {\mu_w}\right)^{0.14}

    .. math::
        Nu_J = \frac{h d_{ch}}{k}

    .. math::
        Nu_A = 3.66

    .. math::
        Nu_B = 1.62 Pr^{1/3}Re_{J,eq}^{1/3}\left(\frac{d_{ch}}{l_{ch}}
        \right)^{1/3}

    .. math::
        Nu_C = 0.664Pr^{1/3}(Re_{J,eq}\frac{d_{ch}}{l_{ch}})^{0.5}

    .. math::
        \text{if } Re_{J,eq} < 2300: Nu_D = 0

    .. math::
        Nu_D = 0.0115Pr^{1/3}Re_{J,eq}^{0.9}\left(1 - \left(\frac{2300}
        {Re_{J,eq}}\right)^{2.5}\right)\left(1 + \left(\frac{d_{ch}}{l_{ch}}
        \right)^{2/3}\right)


    For Radial inlets:

    .. math::
        v_{ch} = v_{Mit}\left(\frac{\ln\frac{b_{Mit}}{b_{Ein}}}{1 -
        \frac{b_{Ein}}{b_{Mit}}}\right)

    .. math::
        b_{Ein} = \frac{\pi}{8}\frac{D_{inlet}^2}{\delta}

    .. math::
        b_{Mit} = \frac{\pi}{2}D_{tank}\sqrt{1 + \frac{\pi^2}{4}\frac
        {D_{tank}^2}{H^2}}

    .. math::
        v_{Mit} = \frac{Q}{2\delta b_{Mit}}

    For Tangential inlets:

    .. math::
        v_{ch} = (v_x^2 + v_z^2)^{0.5}

    .. math::
        v_x = v_{inlet}\left(\frac{\ln[1 + \frac{f_d D_{tank}H}{D_{inlet}^2}
        \frac{v_x(0)}{v_{inlet}}]}{\frac{f_d D_{tank}H}{D_{inlet}^2}}\right)

    .. math::
        v_x(0) = K_3 + (K_3^2 + K_4)^{0.5}

    .. math::
        K_3 = \frac{v_{inlet}}{4} -\frac{D_{inlet}^2v_{inlet}}{4f_d D_{tank}H}

    .. math::
        K_4 = \frac{D_{inlet}^2v_{inlet}^2}{2f_d D_{tank} H}

    .. math::
        v_z = \frac{Q}{\pi D_{tank}\delta}

    .. math::
        v_{inlet} = \frac{Q}{\frac{\pi}{4}D_{inlet}^2}


    Parameters
    ----------
    m : float
        Mass flow rate of fluid, [kg/m^3]
    Dtank : float
        Outer diameter of tank or vessel surrounded by jacket, [m]
    Djacket : float
        Inner diameter of jacket surrounding a vessel or tank, [m]
    H : float
        Height of the vessel or tank, [m]
    Dinlet : float
        Inner diameter of inlet into the jacket, [m]
    rho : float
        Density of the fluid at Tm [kg/m^3]
    Cp : float
        Heat capacity of fluid at Tm [J/kg/K]
    k : float
        Thermal conductivity of fluid at Tm [W/m/K]
    mu : float
        Viscosity of fluid at Tm [Pa*s]
    muw : float, optional
        Viscosity of fluid at Tw [Pa*s]
    rhow : float, optional
        Density of the fluid at Tw [kg/m^3]
    inlettype : str, optional
        Either 'tangential' or 'radial'
    inletlocation : str, optional
        Either 'top' or 'bottom' or 'auto'
    roughness : float, optional
        Roughness of the tank walls [m]

    Returns
    -------
    h : float
        Average  transfer coefficient inside the jacket [W/m^2/K]

    Notes
    -----
    [1]_ is in German and has not been reviewed. Multiple other formulations
    are considered in [1]_.

    If the fluid is heated and enters from the bottom, natural convection
    assists the heat transfer and the Grashof term is added; if it were to enter
    from the top, it would be subtracted. The situation is reversed if entry
    is from the top.

    Examples
    --------
    Example as in [2]_, matches in all but friction factor:

    >>> Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025,
    ... rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8)
    5695.2041698088615

    References
    ----------
    .. [1] Stein, Prof Dr-Ing Werner Alexander, and Dipl-Ing (FH) Wolfgang
       Schmidt. "Wärmeübergang auf der Wärmeträgerseite eines Rührbehälters mit
       einem einfachen Mantel." Forschung im Ingenieurwesen 59, no. 5
       (May 1993): 73-90. doi:10.1007/BF02561203.
    .. [2] Gesellschaft, V. D. I., ed. VDI Heat Atlas. 2nd edition.
       Berlin; New York:: Springer, 2010.
    '''
    delta = (Djacket-Dtank)/2.
    Q = m/rho
    Pr = Cp*mu/k
    lch = (pi**2/4*Dtank**2 + H**2)**0.5
    dch = 2*delta
    if inlettype == 'radial':
        bEin = pi/8*Dinlet**2/delta
        bMit = pi/2*Dtank*(1 + pi**2/4*Dtank**2/H**2)**0.5
        vMit = Q/(2*delta*bMit)
        vch = vMit*log(bMit/bEin)/(1 - bEin/bMit)
        ReJ = vch*dch*rho/mu
    elif inlettype == 'tangential':
        f = friction_factor(1E5, roughness/dch)
        for run in range(5):
            vinlet = Q/(pi/4*Dinlet**2)
            vz = Q/(pi*Dtank*delta)
            K4 = Dinlet**2*vinlet**2/(2*f*Dtank*H)
            K3 = vinlet/4. - Dinlet**2*vinlet/(4*f*Dtank*H)
            vx0 = K3 + (K3**2 + K4)**0.5
            vx = vinlet*log(1 + f*Dtank*H/Dinlet**2*vx0/vinlet)/(f*Dtank*H/Dinlet**2)
            vch = (vx**2 + vz**2)**0.5
            ReJ = vch*dch*rho/mu
            f = friction_factor(ReJ, roughness/dch)
    if inletlocation and rhow:
        GrJ = g*rho*(rho-rhow)*dch**3/mu**2
        if rhow < rho: # Heating jacket fluid
            if inletlocation == 'auto' or inletlocation == 'bottom':
                ReJeq = (ReJ**2 + GrJ*H/dch/50.)**0.5
            else:
                ReJeq = (ReJ**2 - GrJ*H/dch/50.)**0.5
        else: # Cooling jacket fluid
            if inletlocation == 'auto' or inletlocation == 'top':
                ReJeq = (ReJ**2 + GrJ*H/dch/50.)**0.5
            else:
                ReJeq = (ReJ**2 - GrJ*H/dch/50.)**0.5
    else:
        ReJeq = (ReJ**2)**0.5
    NuA = 3.66
    NuB = 1.62*Pr**(1/3.)*ReJeq**(1/3.)*(dch/lch)**(1/3.)
    NuC = 0.664*Pr**(1/3.)*(ReJeq*dch/lch)**0.5
    if ReJeq < 2300:
        NuD = 0
    else:
        NuD = 0.0115*Pr**(1/3.)*ReJeq**0.9*(1 - (2300./ReJeq)**2.5)*(1 + (dch/lch)**(2/3.))
    if muw:
        NuJ = (NuA**3 + NuB**3 + NuC**3 + NuD**3)**(1/3.)*(mu/muw)**0.14
    else:
        NuJ = (NuA**3 + NuB**3 + NuC**3 + NuD**3)**(1/3.)
    return NuJ*k/dch
