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
from math import pi, log
from scipy.constants import g
from fluids.friction import friction_factor

__all__ =['Lehrer', 'Stein_Schmidt']

def Lehrer(m=None, Dtank=None, Djacket=None, H=None, Dinlet=None,
           rho=None, Cp=None, k=None, mu=None, muw=None,
           isobaric_expansion=None, dT=None, inlettype='tangential',
           inletlocation='auto'):
    r'''Calculates average heat transfer coefficient for a jacket around a
    vessel according to [1]_ as described in [2]_.

    .. math::
        Nu_{S,L} = \left[\frac{0.03Re_S^{0.75}Pr}{1 + \frac{1.74(Pr-1)}
        {Re_S^{0.125}}}\right]\left(\frac{\mu}{\mu_w}\right)^{0.14}

        d_g = \left(\frac{8}{3}\right)^{0.5}\delta

        v_h = (v_Sv_{inlet})^{0.5} + v_A

        v_{inlet} = \frac{Q}{\frac{\pi}{4}d_{inlet}^2}

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
    isobaric_expansion : float, optional
        Constant pressure expansivity of a fluid, [m^3/mol/K]
    dT : float, optional
        Temperature difference of fluid in jacket, K
    inlettype : str, optional
        Either 'tangential' or 'radial'
    inletlocation : str, optional
        Either 'top' or 'bottom' or 'auto'

    Returns
    -------
    h: float
        Average  transfer coefficient inside the jacket [W/m^2/K]

    Notes
    -----
    If the fluid is heated and enters from the bottom, natural convection
    assists the heat tansfer and the Grashof term is added; if it were to enter
    from the top, it would be substracted. The situation is reversed if entry
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
    if dT and isobaric_expansion and inlettype == 'radial' and inletlocation:
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
    if muw:
        NuSL = (0.03*Res**0.75*Pr)/(1 + 1.74*(Pr-1)/Res**0.125)*(mu/muw)**0.14
    else:
        NuSL = (0.03*Res**0.75*Pr)/(1 + 1.74*(Pr-1)/Res**0.125)
    h = NuSL*k/dg
    return h


def Stein_Schmidt(m=None, Dtank=None, Djacket=None, H=None, Dinlet=None,
                  rho=None, Cp=None, k=None, mu=None, muw=None, rhow=None,
                  inlettype='tangential', inletlocation='auto', roughness=0):
    r'''Calculates average heat transfer coefficient for a jacket around a
    vessel according to [1]_ as described in [2]_.

    .. math::
        l_{ch} = \left[\left(\frac{\pi}{2}\right)^2 D_{tank}^2+H^2\right]^{0.5}

        d_{ch} = 2\delta

        Re_j = \frac{v_{ch}d_{ch}\rho}{\mu}

        Gr_J = \frac{g\rho(\rho-\rho_w)d_{ch}^3}{\mu^2}

        Re_{J,eq} = \left[Re_J^2\pm \left(\frac{|Gr_J|\frac{H}{d_{ch}}}{50}
        \right)\right]^{0.5}

        Nu_J = (Nu_A^3 + Nu_B^3 + Nu_C^3 + Nu_D^3)^{1/3}\left(\frac{\mu}
        {\mu_w}\right)^{0.14}

        Nu_J = \frac{h d_{ch}}{k}

        Nu_A = 3.66

        Nu_B = 1.62 Pr^{1/3}Re_{J,eq}^{1/3}\left(\frac{d_{ch}}{l_{ch}}
        \right)^{1/3}

        Nu_C = 0.664Pr^{1/3}(Re_{J,eq}\frac{d_{ch}}{l_{ch}})^{0.5}

        \text{if } Re_{J,eq} < 2300: Nu_D = 0

        Nu_D = 0.0115Pr^{1/3}Re_{J,eq}^{0.9}\left(1 - \left(\frac{2300}
        {Re_{J,eq}}\right)^{2.5}\right)\left(1 + \left(\frac{d_{ch}}{l_{ch}}
        \right)^{2/3}\right)


    For Radial inlets:

    .. math::
        v_{ch} = v_{Mit}\left(\frac{\ln\frac{b_{Mit}}{b_{Ein}}}{1 -
        \frac{b_{Ein}}{b_{Mit}}}\right)

        b_{Ein} = \frac{\pi}{8}\frac{D_{inlet}^2}{\delta}

        b_{Mit} = \frac{\pi}{2}D_{tank}\sqrt{1 + \frac{\pi^2}{4}\frac
        {D_{tank}^2}{H^2}}

        v_{Mit} = \frac{Q}{2\delta b_{Mit}}

    For Tangential inlets:

    .. math::
        v_{ch} = (v_x^2 + v_z^2)^{0.5}

        v_x = v_{inlet}\left(\frac{\ln[1 + \frac{f_d D_{tank}H}{D_{inlet}^2}
        \frac{v_x(0)}{v_{inlet}}]}{\frac{f_d D_{tank}H}{D_{inlet}^2}}\right)

        v_x(0) = K_3 + (K_3^2 + K_4)^{0.5}

        K_3 = \frac{v_{inlet}}{4} -\frac{D_{inlet}^2v_{inlet}}{4f_d D_{tank}H}

        K_4 = \frac{D_{inlet}^2v_{inlet}^2}{2f_d D_{tank} H}

        v_z = \frac{Q}{\pi D_{tank}\delta}

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
    h: float
        Average  transfer coefficient inside the jacket [W/m^2/K]

    Notes
    -----
    [1]_ is in German and has not been reviewed. Multiple other formulations
    are considered in [1]_.

    If the fluid is heated and enters from the bottom, natural convection
    assists the heat tansfer and the Grashof term is added; if it were to enter
    from the top, it would be substracted. The situation is reversed if entry
    is from the top.

    Examples
    --------
    Example as in [2]_, matches in all but friction factor:

    >>> Stein_Schmidt(m=2.5, Dtank=0.6, Djacket=0.65, H=0.6, Dinlet=0.025,
    ... rho=995.7, Cp=4178.1, k=0.615, mu=798E-6, muw=355E-6, rhow=971.8)
    5695.1871940874225

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
    h = NuJ*k/dch
    return h
# Eveything here is good.
