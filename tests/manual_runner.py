#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
try:
    import test_air_cooler
except:
    print('run this from the tests directory')
    exit()
import test_boiling_flow
import test_boiling_nucleic
import test_boiling_plate
import test_condensation
import test_conduction
import test_conv_external
import test_conv_free_immersed
import test_conv_free_enclosed
import test_conv_internal
import test_conv_jacket
import test_conv_packed_bed
import test_conv_plate
import test_conv_supercritical
import test_conv_tube_bank
import test_conv_two_phase
import test_core
import test_hx
import test_radiation
# dynamically generated code - numba, units, vectorize - not part of this test suite
to_test = [test_air_cooler, test_boiling_flow, test_boiling_nucleic, test_boiling_plate, test_condensation, test_conduction, test_conv_external, test_conv_free_immersed, test_conv_free_enclosed, test_conv_internal, test_conv_jacket, test_conv_packed_bed, test_conv_plate, test_conv_supercritical, test_conv_tube_bank, test_conv_two_phase, test_core, test_hx, test_radiation]


skip_marks = ['slow', 'fuzz', 'skip_types']
skip_marks_set = set(skip_marks)
if len(sys.argv) >= 2:
    #print(sys.argv)
    # Run modules specified by user
    to_test = [globals()[i] for i in sys.argv[1:]]
for mod in to_test:
    print(mod)
    for s in dir(mod):
        skip = False
        obj = getattr(mod, s)
        if callable(obj) and hasattr(obj, '__name__') and obj.__name__.startswith('test'):
            try:
                for bad in skip_marks:
                    if bad in obj.__dict__:
                        skip = True
                if 'pytestmark' in obj.__dict__:
                    marked_names = [i.name for i in obj.__dict__['pytestmark']]
                    for mark_name in marked_names:
                        if mark_name in skip_marks_set:
                            skip = True
            except Exception as e:
                #print(e)
                pass
            if not skip:
                try:
                    print(obj)
                    obj()
                except Exception as e:
                    print('FAILED TEST %s with error:' %s)
                    print(e)
