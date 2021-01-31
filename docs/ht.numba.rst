Support for Numba (ht.numba)
============================

Basic module which wraps most of ht functions and classes to be compatible with the
`Numba <https://github.com/numba/numba>`_ dynamic Python compiler.
Numba is only supported on Python 3, and may require the latest version of Numba.
Numba is rapidly evolving, and hopefully in the future it will support more of
the functionality of ht.

Using the numba-accelerated version of `ht` is easy; simply call functions
and classes from the ht.numba namespace. The ht.numba module must be
imported separately; it is not loaded automatically as part of ht.

>>> import ht
>>> import ht.numba
>>> ht.numba.Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4)
0.55050936040

There is a delay while the code is compiled when using Numba;
the speed is not quite free. Most, but not all compilations can be
cached to save time in future loadings.

It is easy to compare the speed of a function with and without Numba.

>>> %timeit ht.numba.Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4) # doctest: +SKIP
1.22 µs ± 41.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
>>> %timeit ht.Ft_aircooler(Thi=125., Tho=45., Tci=25., Tco=95., Ntp=1, rows=4) # doctest: +SKIP
5.89 µs ± 274 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Not everything is faster in the numba interface. It is advisable to check 
that numba is indeed faster for your use case.

Functions which take strings as inputs are also known to normally get slower;
the numerical stuff is still being sped up but the string handling is slow:

>>> %timeit ht.numba.baffle_correction_Bell(0.82, method='spline') # doctest: +SKIP
16.5 µs ± 538 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
>>> %timeit ht.baffle_correction_Bell(0.82, method='spline') # doctest: +SKIP
15.6 µs ± 457 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

Nevertheless, using the function from the numba interface may be preferably,
to allow an even larger program to be completely compiled in njit mode.

Today, the list of things known not to work is as follows:

- :py:func:`~.dP_Zukauskas` (needs some spline work)
- :py:func:`~.cylindrical_heat_transfer` (returns dictionaries)
- :py:func:`~.effectiveness_NTU_method` (returns dictionaries)
- :py:func:`~.P_NTU_method` (returns dictionaries)
- :py:func:`~.NTU_from_effectiveness` (does string-to-int conversion)
- :py:func:`~.DBundle_min` and :py:func:`~.shell_clearance` (needs work)
- :py:func:`~.wall_factor_Nu` and :py:func:`~.wall_factor_fd` (dictionary lookups)
- :py:func:`~.solar_spectrum` (external file reading)
- Everything in :py:mod:`ht.insulation`


Numpy Support
-------------
Numba also allows ht to provide any of its supported functions as a numpy universal
function. Numpy's wonderful broadcasting is implemented, so some arguments can
be arrays and some can not.

>>> import ht.numba_vectorized
>>> import numpy as np
>>> ht.numba_vectorized.Nu_Grimison_tube_bank(np.linspace(1e4, 1e5, 4), np.array([.708]), np.array([11]), np.array([.05]), np.array([.05]), np.array([.025]))
array([3.39729780e+06, 3.74551216e+07, 9.86950909e+07, 1.83014426e+08])

Unfortunately, keyword-arguments are not supported by Numba.
Also default arguments are not presently supported by Numba.

Despite these limitations is is here that Numba really shines! Arrays are Numba's
strength.

Please note this interface is provided, but what works and what doesn't is
mostly up to the numba project. This backend is not quite as polished as
their normal engine.
