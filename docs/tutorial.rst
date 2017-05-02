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

