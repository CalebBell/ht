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

__all__ = ['Nu_McAdams', 'Nu_Shitsman', 'Nu_Griem', 'Nu_Jackson']


def Nu_McAdams(Re, Pr):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in a pipe under supercritical conditions according to [1]_.
    
    Found in [2]_ to fit the enhanced heat transfer regime with a MAD of 10.3%
    which was better than and of the other reviewed correlations.
    
    .. math::
        Nu_b = 0.0243*Re_b^{0.8}Pr_b^{0.4}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    This has also been one of the forms of the Dittus-Boelter correlations. 
    Claimed to fit data for high pressures and low heat fluxes.
    

    Examples
    --------
    >>> Nu_McAdams(1E5, 1.2)
    261.3838629346147

    References
    ----------
    .. [1] Mac Adams, William H. Heat Transmission. New York and London, 1942.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at 
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015): 
       451-60. doi:10.1016/j.anucene.2014.10.027.
    '''
    return 0.0243*Re**0.8*Pr**0.4


def Nu_Shitsman(Re, Prb, Prw):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in a pipe under supercritical conditions according to [1]_ and [2] as shown
    in both [3]_ and [4]_.
        
    .. math::
        Nu_b = 0.023 Re_b^{0.8}(min(Pr_b, Pr_w))^{0.8}

    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Prb : float
        Prandtl number with bulk fluid properties, [-]
    Prw : float
        Prandtl number with wall fluid properties, [-]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    [3]_ states this correlation was developed with D = 7.8 and 8.2 mm and with
    a `Pr` approximately 1. [3]_ ranked it third in the enhanced heat transfer
    category, with a MAD as 11.5%
    
    [4]_ cites a [1]_ as the source of the correlation. Neither have been
    reviewd, and both are in Russian. [4]_ lists this as third most accurate
    of the 14 correlations reviewed from a database of all regimes.

    Examples
    --------
    >>> Nu_Shitsman(1E5, 1.2, 1.6)
    266.1171311047253

    References
    ----------
    .. [1] M. E Shitsman, Impairment of the heat transmission at supercritical 
       pressures, High. Temperature, 1963, 1(2): 237-244
    .. [2] Miropol’skiy ZL, Shitsman ME (1957). Heat transfer to water and 
       steam at variable specific heat. J Tech Phys XXVII(10): 2359-2372
    .. [3] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at 
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015): 
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [4] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of 
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using 
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009): 
       757-66. doi:10.1007/s00231-008-0475-4.
    '''
    return 0.023*Re**0.8*min(Prb, Prw)**0.8
    

def Nu_Griem(Re, Pr, H=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in a pipe under supercritical conditions according to [1]_, also shown in
    [2]_, [3]_ and [4]_. Has complicated rules regarding where properties 
    should be evaluated.
        
    .. math::
        Nu_m = 0.0169Re_b^{0.8356} Pr_{sel}^{0.432}\omega

    Parameters
    ----------
    Re : float
        Reynolds number as explained below, [-]
    Pr : float
        Prandtl number as explained below, [-]
    H : float
        Enthalpy of water (if the fluid is water), [J/kg]

    Returns
    -------
    Nu : float
        Nusselt number as explained below, [-]

    Notes
    -----
    w is calculated as follows, for water only, with a reference point from
    the 1967-IFC formulation. It is set to 1 if H is not provided:
    if Hb < 1.54E6 J/kg, w = 0.82; if Hb > 1.74E6 J/kg, w = 1; otherwise 
    w = 0.82 + 9E-7*(Hb-1.54E6). 
    
    To determine heat capacity to be used, Cp should be calculated at 5 points, 
    and the lowest three points should be averaged.
    The five points are: Tw, (Tw+Tf)/2, Tf, (Tb+Tf)/2, Tb.
    
    Viscosity should be the bulk viscosity.
    Thermal conductivity should be the average of the bulk and wall values.
    Density should be the bulk density. 
    
    [2]_ states this correlation was developed with D = 10, 14, and 20 mm,
    P from 22 to 27 MPa, G from 300 to 2500 kg/m^2/s, and q from 
    200 to 700 kW/m^2. It was ranked 6th among the 14 correlations reviewed for
    enhanced heat transfer, with a MAD of 13.8%, and 6th overall for the three
    heat transfer conditions with a overall MAD of 14.8%. [3]_ ranked it 8th
    of 14 correlations for the three heat transfer conditions.
    
    [2]_ has an almost complete description of the model; both [3]_ and [4]_
    simplify it.

    Examples
    --------
    >>> Nu_Griem(1E5, 1.2)
    275.4818576600527

    References
    ----------
    .. [1] Griem, H. "A New Procedure for the Prediction of Forced Convection 
       Heat Transfer at near- and Supercritical Pressure." Heat and Mass 
       Transfer 31, no. 5 (1996): 301-5. doi:10.1007/BF02184042.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at 
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015): 
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of 
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using 
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009): 
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado. 
       "Review and Proposal for Heat Transfer Predictions at Supercritical 
       Water Conditions Using Existing Correlations and Experiments." Nuclear 
       Engineering and Design, (W3MDM) University of Leeds International 
       Symposium: What Where When? Multi-dimensional Advances for Industrial 
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203. 
       doi:10.1016/j.nucengdes.2011.03.022. 
    '''
    if H and H < 1.54E6:
        w = 0.82
    elif H and H > 1.74E6:
        w = 1
    elif H:
        w = 0.82 + 9E-7*(H - 1.54E6)
    else:
        w = 1
    Nu = 0.0169*Re**0.8356*Pr**0.432*w
    return Nu


def Nu_Jackson(Re, Pr, rho_w=None, rho_b=None, Cp_avg=None, Cp_b=None, T_b=None,
               T_w=None, T_pc=None):
    r'''Calculates internal convection Nusselt number for turbulent flows
    in a pipe under supercritical conditions according to [1]_.
        
    .. math::
        Nu_b = 0.0183 Re_b^{0.82} Pr^{0.5}
        \left(\frac{\rho_w}{\rho_b}\right)^{0.3}
        \left(\frac{\bar C_p}{C_{p,b}}\right)^n
        
        n = 0.4 \text{ for } T_b < T_w < T_{pc} \text{ or } 1.2T_{pc} < T_b < T_w
        
        n = 0.4 + 0.2(T_w/T_{pc} - 1) \text{ for } T_b < T_{pc} <T_w
        
        n = 0.4 + 0.2(T_w/T_{pc} - 1)[1 - 5(T_b/T_{pc}-1)]
        \text{ for } T_{pc} < T_b < 1.2T_{pc} \text{ and } T_b < T_w

        \bar{Cp} = \frac{H_w-H_b}{T_w-T_b}
        
    Parameters
    ----------
    Re : float
        Reynolds number with bulk fluid properties, [-]
    Pr : float
        Prandtl number with bulk fluid properties, [-]
    rho_w : float, optional
        Density at the wall temperature, [kg/m^3]
    rho_b : float, optional
        Density at the bulk temperature, [kg/m^3]
    Cp_avg : float, optional
        Average heat capacity between the wall and bulk temperatures, [J/kg/K]
    Cp_b : float, optional
        Heat capacity at the bulk temperature, [J/kg/K]
    T_b : float
        Bulk temperature, [K]
    T_w : float
        Wall temperature, [K]
    T_pc : float
        Pseudocritical temperature, i.e. temperature at P where Cp is at a 
        maximum, [K]

    Returns
    -------
    Nu : float
        Nusselt number with bulk fluid properties, [-]

    Notes
    -----
    The range of examined paramters is as follows: 
    P from 23.4 to 29.3 MPa; G from 700-3600 kg/m^2/s; 
    q from 46 to 2600 kW/m^2; Re from 8E4 to 5E5; D from 1.6 to 20 mm.
    
    For enhanced heat transfer database in [2]_, this correlation was the
    second best with a MAD of 11.5%. In the database in [3]_, the correlation
    was the second best as well.
    
    This is sometimes called the Jackson-Hall correlation.
    If the extra information is not provided, the correlation will be used
    without the corrections.

    Examples
    --------
    >>> Nu_Jackson(1E5, 1.2)
    252.37231572974918

    References
    ----------
    .. [1] Jackson, J. D. "Consideration of the Heat Transfer Properties of 
       Supercritical Pressure Water in Connection with the Cooling of Advanced 
       Nuclear Reactors", 2002.
    .. [2] Chen, Weiwei, Xiande Fang, Yu Xu, and Xianghui Su. "An Assessment of
       Correlations of Forced Convection Heat Transfer to Water at 
       Supercritical Pressure." Annals of Nuclear Energy 76 (February 2015): 
       451-60. doi:10.1016/j.anucene.2014.10.027.
    .. [3] Yu, Jiyang, Baoshan Jia, Dan Wu, and Daling Wang. "Optimization of 
       Heat Transfer Coefficient Correlation at Supercritical Pressure Using 
       Genetic Algorithms." Heat and Mass Transfer 45, no. 6 (January 8, 2009): 
       757-66. doi:10.1007/s00231-008-0475-4.
    .. [4] Jäger, Wadim, Victor Hugo Sánchez Espinoza, and Antonio Hurtado. 
       "Review and Proposal for Heat Transfer Predictions at Supercritical 
       Water Conditions Using Existing Correlations and Experiments." Nuclear 
       Engineering and Design, (W3MDM) University of Leeds International 
       Symposium: What Where When? Multi-dimensional Advances for Industrial 
       Process Monitoring, 241, no. 6 (June 2011): 2184-2203. 
       doi:10.1016/j.nucengdes.2011.03.022. 
    '''
    if all([T_b, T_w, T_pc]):
        if T_b < T_w < T_pc or 1.2*T_pc < T_b < T_w:      
            n = 0.4
        elif T_b < T_pc < T_w:
            n = 0.4 + 0.2*(T_w/T_pc - 1)
        else:
            n = 0.4 + 0.2*(T_w/T_pc - 1)*(1 - 5*(T_b/T_pc - 1))
    else:
        n = 0.4
    Nu = 0.0183*Re**0.82*Pr**0.5
    if rho_w and rho_b:
        Nu *= (rho_w/rho_b)**0.3
    if Cp_avg and Cp_b:
        Nu *= (Cp_avg/Cp_b)**n
    return Nu

