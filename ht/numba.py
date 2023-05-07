'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2020, 2021, 2022, 2023 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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

import inspect

import fluids
import fluids.numba
import numba

import ht

normal_fluids = fluids
normal = ht

orig_file = __file__
caching = False
"""


"""
__all__ = []
__funcs = {}

numerics = fluids.numba.numerics
replaced = fluids.numba.numerics_dict.copy()


def transform_complete_ht(replaced, __funcs, __all__, normal, vec=False):
    cache_blacklist = {'h_Ganguli_VDI', 'fin_efficiency_Kern_Kraus', 'h_Briggs_Young',
                           'h_ESDU_high_fin', 'h_ESDU_low_fin', 'Nu_Nusselt_Rayleigh_Holling_Herwig',
                           'DBundle_for_Ntubes_Phadkeb', 'Thome', 'to_solve_q_Thome',
                       'temperature_effectiveness_air_cooler', 'factorial',
                       'size_bundle_from_tubecount', 'crossflow_effectiveness_to_int',
                       'temperature_effectiveness_basic', '_NTU_from_P_solver',
                       'NTU_from_P_basic', '_NTU_from_P_erf',
                       'NTU_from_P_G', 'NTU_from_P_J', 'NTU_from_P_E',
                       'NTU_from_P_H', 'NTU_from_P_plate',
                       '_NTU_from_P_objective',
                       'temperature_effectiveness_plate', # dies on recursion
                       }
    __funcs.update(normal_fluids.numba.numbafied_fluids_functions.copy())
    new_mods = normal_fluids.numba.transform_module(normal, __funcs, replaced, vec=vec,
                                                    cache_blacklist=cache_blacklist)
    if vec:
        conv_fun = numba.vectorize
    else:
        conv_fun = numba.njit

    to_change_full_output = []

    to_change = {}
    to_change.update({k: 'full_output' for k in to_change_full_output})
#    to_change['hx.Ntubes_Phadkeb'] = 'square_C1s is None'
    to_change['boiling_nucleic.Gorenflo'] = 'h0 is None: # NUMBA: DELETE'

    for s, bad_branch in to_change.items():
        mod, func = s.split('.')
        source = inspect.getsource(getattr(getattr(normal, mod), func))
        fake_mod = __funcs[mod]
        source = normal_fluids.numba.remove_branch(source, bad_branch)
        normal_fluids.numba.numba_exec_cacheable(source, fake_mod.__dict__, fake_mod.__dict__)
        new_func = fake_mod.__dict__[func]
        obj = conv_fun(cache=caching)(new_func)
        __funcs[func] = obj
        globals()[func] = obj
        obj.__doc__ = ''
    to_change = ['air_cooler.Ft_aircooler', 'hx.Ntubes_Phadkeb',
                 'hx.DBundle_for_Ntubes_Phadkeb', 'boiling_nucleic.h_nucleic_methods',
                 'hx._NTU_from_P_solver', 'hx.NTU_from_P_plate']
    normal_fluids.numba.transform_lists_to_arrays(normal, to_change, __funcs, cache_blacklist=cache_blacklist)

    for mod in new_mods:
        mod.__dict__.update(__funcs)
        try:
            __all__.extend(mod.__all__)
        except AttributeError:
            pass

    __funcs['hx']._load_coeffs_Phadkeb() # Run after everything is done

transform_complete_ht(replaced, __funcs, __all__, normal, vec=False)



globals().update(__funcs)
globals().update(replaced)

__name__ = 'ht.numba'
__file__ = orig_file
