Tutorial
========

Introduction
------------

Log mean temperature are available for both counterflow (by default) and 
co-current flow. This calculation does not depend on the units of temperature
provided.

>>> LMTD(Thi=100, Tho=60, Tci=30, Tco=40.2)
43.200409294131525
>>> LMTD(100, 60, 30, 40.2, counterflow=False)
39.75251118049003

Design philosophy
-----------------
Like all libraries, this was developed to scratch my own itches. Since its
public release it has been found useful by many others, from students across 
the world to practicing engineers at some of the world's largest companies.

The bulk of this library's API is considered stable; enhancements to 
functions and classes will still happen, and default methods when using a generic 
correlation interface may change to newer and more accurate correlations as
they are published and reviewed.

To the extent possible, correlations are implemented depending on the highest
level parameters. The Nu_conv_internal correlation does not accept pipe diameter,
velocity, viscosity, density, heat capacity, and thermal conductivity - it accepts 
Reynolds number and Prandtl number. This makes the API cleaner and encourages modular design.

All functions are desiged to accept inputs in base SI units. However, any 
set of consistent units given to a function will return a consistent result;
for instance, a function calculating volume doesn't care if given an input in
inches or meters; the output units will be the cube of those given to it.
The user is directed to unit conversion libraries such as 
`pint <https://github.com/hgrecco/pint>`_ to perform unit conversions if they
prefer not to work in SI units.

The standard math library is used in all functions except where special
functions from numpy or scipy are necessary. SciPy is used for root finding,
interpolation, scientific constants, ode integration, and its many special
mathematical functions not present in the standard math library. The only other
required library is the `fluids` library, a sister library for fluid dynamics.
No other libraries will become required dependencies; anything else is optional.

To allow use of numpy arrays with ht, a `vectorized` module is implemented,
which wraps all of the ht functions with np.vectorize. Instead of importing
from ht, the user can import from ht.vectorized:

>>> from ht.vectorized import *
>>> LMTD([100, 101], 60., 30., 40.2)
array([ 43.20040929,  43.60182765])


Insulation
----------

Insulating and refractory materials from the VDI Heat Transfer Handbook
and the ASHRAE Handbook: Fundamentals have been digitized and are programatically
available in ht. Density, heat capacity, and thermal conductivity are available
although not all materials have all three.

The actual data is stored in a series of dictionaries, building_materials, 
ASHRAE_board_siding, ASHRAE_flooring, ASHRAE_insulation, ASHRAE_roofing, 
ASHRAE_plastering, ASHRAE_masonry, ASHRAE_woods, and refractories.
A total of 390 different materials are available.
Functions have been written to make accessing this data much 
more convenient. 

To determine the correct string to look up a material by, one can use the
function nearest_material:

>>> nearest_material('stainless steel')
'Metals, stainless steel'
>>> nearest_material('mineral fibre')
'Mineral fiber'

Knowing a material's ID, the functions k_material, rho_material, and Cp_material
can be used to obtain its properties.

>>> wood = nearest_material('spruce')
>>> k_material(wood)
0.09
>>> rho_material(wood)
400.0
>>> Cp_material(wood)
1630.0

Materials which are refractories, stored in the dictionary `refractories`,
have temperature dependent heat capacity and thermal conductivity between
400 °C and 1200 °C.

>>> C = nearest_material('graphite')
>>> k_material(C)
67.0
>>> k_material(C, T=800)
62.9851975

The limiting values are returned outside of this range:

>>> Cp_material(C, T=8000), Cp_material(C, T=1)
(1588.0, 1108.0)


Radiation
---------
The Stefan-Boltzman law is implemented as `q_rad`. Optionally, a surrounding
temperature may be specified as well. If the surrounding temperature is higher
than the object, the calculated heat flux in W/m^2 will be negative, indicating
the object is picking up heat not losing it.

>>> q_rad(emissivity=1, T=400)
1451.613952
>>> q_rad(.85, T=400, T2=305.)
816.7821722650002
>>> q_rad(.85, T=400, T2=5000) # ouch
-30122590.815640796

A blackbody's spectral radiance can also be calculated, in units of 
W/steradian/square metre/metre. This calculation requires the temperature
of the object and the wavelength to be considered.

>>> blackbody_spectral_radiance(T=800., wavelength=4E-6)
1311692056.2430143

Heat exchanger sizing
---------------------

There are three popular methods of sizing heat exchangers. The log-mean temperature 
difference correction factor method, the ε-NTU method, and the P-NTU method.
Each of those are cannot size a heat exchanger on their own - they do not
care about heat transfer coefficients or area - but they must be used first
to determine the thermal conditions of the heat exchanger. Sizing a heat exchanger
is a very iterative process, and many designs should be attempted to determine
the optimal one based on required performance and cost. The P-NTU method
supports the most types of heat exchangers; its form always requires the UA
term to be guessed however.


LMTD correction factor method
-----------------------------

The simplest method, the log-mean temperature difference correction factor method,
is as follows:

.. math::
    Q = UA\Delta T_{lm} F_t
    
Knowing the outlet and inlet temperatures of a heat exchanger and `Q`, one could
determine `UA` as follows:

>>> dTlm = LMTD(Tci=15, Tco=85, Thi=130, Tho=110)
>>> Ft = F_LMTD_Fakheri(Tci=15, Tco=85, Thi=130, Tho=110, shells=1)
>>> Q = 1E6 # 1 MW
>>> UA = Q/(dTlm*Ft)
>>> UA
15833.566307803789

This method requires you to know all four temperatures before UA can be calculated.
Fakheri developed a general expression for calculating `Ft`; it is valid for
counterflow shell-and-tube exchangers with an even number of tube passes; the 
number of shell-side passes can be varied. `Ft` is always less than 1, approaching
1 with very high numbers of shells:

>>> F_LMTD_Fakheri(Tci=15, Tco=85, Thi=130, Tho=110, shells=10)
0.9994785295070708

No other expressions are available to calculate `Ft` for different heat exchanger
geometries; only the TEMA F and E exchanger types are really covered by this 
expression. However, with results from the other methods, `Ft` can always
be back-calculated.

Effectiveness-NTU method
------------------------
This method uses the formula :math:`Q=\epsilon C_{min}(T_{h,i}-T_{c,i})`. The main
complication of this method is calculating effectiveness `epsilon`, which
is a function of the mass flows, heat capacities, and UA
:math:`\epsilon = f(NTU, C_r)`. The effectiveness-NTU method is implemented in 
in `effectiveness_from_NTU` and `NTU_from_effectiveness`. The supported
heat exchanger types are somewhat limited; they are:

* Counterflow (ex. double-pipe)
* Parallel (ex. double pipe inefficient configuration)
* Shell and tube exchangers with even numbers of tube passes,
  one or more shells in series (TEMA E (one pass shell) only)
* Crossflow, single pass, fluids unmixed
* Crossflow, single pass, Cmax mixed, Cmin unmixed
* Crossflow, single pass, Cmin mixed, Cmax unmixed
* Boiler or condenser


To illustrate the method, first the individual methods will be used to 
determine the outlet temperatures of a heat exchanger. After, the
more convenient and flexible wrapper `effectiveness_NTU_method` is
shown. Overall case of rating an existing heat exchanger where a known flowrate
of steam and oil are contacted in crossflow, with the steam side mixed:
    
>>> U = 275 # W/m^2/K
>>> A = 10.82 # m^2
>>> Cp_oil = 1900 # J/kg/K
>>> Cp_steam = 1860 # J/kg/K
>>> m_steam = 5.2 # kg/s
>>> m_oil = 0.725 # kg/s
>>> Thi = 130 # °C
>>> Tci = 15 # °C
>>> Cmin = calc_Cmin(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
>>> Cmax = calc_Cmax(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
>>> Cr = calc_Cr(mh=m_steam, mc=m_oil, Cph=Cp_steam, Cpc=Cp_oil)
>>> NTU = NTU_from_UA(UA=U*A, Cmin=Cmin)
>>> eff = effectiveness_from_NTU(NTU=NTU, Cr=Cr, subtype='crossflow, mixed Cmax')
>>> Q = eff*Cmin*(Thi - Tci)
>>> Tco = Tci + Q/(m_oil*Cp_oil)
>>> Tho = Thi - Q/(m_steam*Cp_steam)
>>> Cmin, Cmax, Cr
(1377.5, 9672.0, 0.14242142266335814)
>>> NTU, eff, Q
(2.160072595281307, 0.8312180361425988, 131675.32715043944)
>>> Tco, Tho
(110.59007415639887, 116.38592564614977)

That was not very convenient. The more helpful wrapper `effectiveness_NTU_method`
needs only the heat capacities and mass flows of each stream, the type of the heat
exchanger, and one combination of the following inputs is required:
        
* Three of the four inlet and outlet stream temperatures
* Temperatures for the cold outlet and hot outlet and UA
* Temperatures for the cold inlet and hot inlet and UA
* Temperatures for the cold inlet and hot outlet and UA
* Temperatures for the cold outlet and hot inlet and UA

The function returns all calculated parameters for convenience as a dictionary.

Solve a heat exchanger to determine UA and effectiveness given the
configuration, flows, subtype, the cold inlet/outlet temperatures, and the
hot stream inlet temperature.

>>> pprint(effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, 
... subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Thi=130))
{'Cmax': 9672.0,
'Cmin': 2755.0,
'Cr': 0.2848428453267163,
'NTU': 1.1040839095588,
'Q': 192850.0,
'Tci': 15,
'Tco': 85,
'Thi': 130,
'Tho': 110.06100082712986,
'UA': 3041.751170834494,
'effectiveness': 0.6086956521739131}

Solve the same heat exchanger with the UA specified, and known inlet
temperatures:
    
>>> pprint(effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, 
... subtype='crossflow, mixed Cmax', Tci=15, Thi=130, UA=3041.75))
{'Cmax': 9672.0,
'Cmin': 2755.0,
'Cr': 0.2848428453267163,
'NTU': 1.1040834845735028,
'Q': 192849.96310220254,
'Tci': 15,
'Tco': 84.99998660697007,
'Thi': 130,
'Tho': 110.06100464203861,
'UA': 3041.75,
'effectiveness': 0.6086955357127832}

