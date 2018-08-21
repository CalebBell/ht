# -*- coding: utf-8 -*-
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
SOFTWARE.'''

from __future__ import division
from math import log, exp, sqrt, tanh, factorial
from ht import *
import numpy as np
from numpy.testing import assert_allclose
import pytest
from random import uniform, randint, seed, choice
seed(0)


def test_Ntubes_Perrys():
    Nt_perry = [[Ntubes_Perrys(DBundle=1.184, Ntp=i, Do=.028, angle=j) for i in [1,2,4,6]] for j in [30, 45, 60, 90]]
    Nt_values = [[1001, 973, 914, 886], [819, 803, 784, 769], [1001, 973, 914, 886], [819, 803, 784, 769]]
    assert_allclose(Nt_perry, Nt_values)
#    angle = 30 or 60 and ntubes = 1.5 raise exception

    with pytest.raises(Exception):
        Ntubes_Perrys(DBundle=1.184, Ntp=5, Do=.028, angle=30)
    with pytest.raises(Exception):
        Ntubes_Perrys(DBundle=1.184, Ntp=5, Do=.028, angle=45)




def test_Ntubes_Phadkeb():
    # For the 45 degree case, ten exanples are given and known to be correct.
    # Unfortunately no examples were given for any other case.
    Ntubes_calc = [Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=i, angle=45.) for i in [1,2,4,6,8]]
    assert_allclose(Ntubes_calc, [805, 782, 760, 698, 680])
    Ntubes_calc = [Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.035, Ntp=i, angle=45.) for i in [1,2,4,6,8]]
    assert_allclose(Ntubes_calc, [861, 838, 816, 750, 732])
    
    # Extra tests
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=2, angle=30.)
    assert N == 898
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=2, angle=60.)
    assert N == 876
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=6, angle=60.)
    assert N == 822
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=8, angle=60.)
    assert N == 772
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.092, Ntp=8, angle=60.)
    assert N == 88
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=8, angle=30.)
    assert N == 788
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.04, Ntp=6, angle=30.)
    assert N == 652
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=8, angle=90.)
    assert N == 684
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=2, angle=90.)
    assert N == 772
    N = Ntubes_Phadkeb(DBundle=1.200-.008*2, Do=.028, pitch=.036, Ntp=6, angle=90.)
    assert N == 712
    
    # Big case
    N = Ntubes_Phadkeb(DBundle=5, Do=.028, pitch=.036, Ntp=2, angle=90.)
    assert N == 14842
    
    # negative case
    N = Ntubes_Phadkeb(DBundle=0.004750018463796297, Do=.001, pitch=.0015, Ntp=8, angle=60)
    assert N == 0
    
    # reverse case
    # DBundle_for_Ntubes_Phadkeb(Ntubes=17546, Do=.001, pitch=.00125, Ntp=6, angle=45) 0.19052937784048926

    with pytest.raises(Exception):
        Ntubes_Phadkeb(DBundle=1.008, Do=.028, pitch=.036, Ntp=11, angle=45.) 
        
    # Test the case of too small for anything
    assert 0 == Ntubes_Phadkeb(DBundle=.01, Do=.028, pitch=.036, Ntp=2, angle=45.) 

def test_Ntubes_Phadkeb_fuzz():
    seed(100)
    D_main = 1E-3
    for angle in [30, 45, 60, 90]:
        for Ntp in [1, 2, 4, 6, 8]:
            for pitch_ratio in [1.25, 1.31, 1.33, 1.375, 1.4, 1.42, 1.5]:
                pitch = D_main*pitch_ratio
                for _ in range(10):
                    DBundle = uniform(pitch*2, pitch*300)
                    N = Ntubes_Phadkeb(DBundle=DBundle, Do=D_main, pitch=pitch, Ntp=Ntp, angle=angle)

    # Test the reverse correlation
    D_main = 1E-2
    for angle in [30, 45, 60, 90]:
        for Ntp in [1, 2, 4, 6, 8]:
            for pitch_ratio in [1.25, 1.31, 1.33, 1.375, 1.4, 1.42, 1.5]:
                pitch = D_main*pitch_ratio
                DBundle = uniform(pitch*5, pitch*300)
                N = Ntubes_Phadkeb(DBundle=DBundle, Do=D_main, pitch=pitch, Ntp=Ntp, angle=angle)
                if N > 2:
                    DBundle2 = DBundle_for_Ntubes_Phadkeb(Ntubes=N, Do=D_main, pitch=pitch, Ntp=Ntp, angle=angle)
                    N2 = Ntubes_Phadkeb(DBundle=DBundle2, Do=D_main, pitch=pitch, Ntp=Ntp, angle=angle)
                    assert N2 == N


@pytest.mark.slow
def test_Phadkeb_numbers():
    # One pain point of this code is that it takes 880 kb to store the results
    # in memory as a list
    from ht.hx import triangular_Ns, triangular_C1s, square_Ns, square_C1s
    from math import floor, ceil
    # Triangular Ns 
    # https://oeis.org/A003136
    # Translated expression originally in Wolfram Mathematica
    # nn = 14; Select[Union[Flatten[Table[x^2 + x*y + y^2, {x, 0, nn}, {y, 0, x}]]], # <= nn^2 &] (* T. D. Noe, Apr 18 2011 *) 
    nums = []
    nn = 400 # Increase this to generate more numbers
    for x in range(0, nn+1):
        for y in range(0, x+1):
            nums.append(x*x + x*y + y*y)
    
    nums = sorted(list(set(nums)))
    
    nn_square = nn*nn
    nums = [i for i in nums if i < nn_square]
    
    nums = nums[0:len(triangular_Ns)]
    assert_allclose(nums, triangular_Ns)
    
    
    # triangular C1s
    # https://oeis.org/A038590 is the sequence, and it is the unique numbers in: 
    # https://oeis.org/A038589
    # Translated from pari expression a(n)=1+6*sum(k=0, n\3, (n\(3*k+1))-(n\(3*k+2)))
    # Tested with the online interpreter http://pari.math.u-bordeaux.fr/gp.html
    # This one is very slow, 300 seconds+
    # Used to be 300 + seconds, now 50+ seconds
    
    def a(n):
        tot = 0
        for k in range(0, int(ceil(n/3.))):
            k3 = k*3.
            tot += floor(n/(k3 + 1.)) - floor(n/(k3 + 2.))
        return 1 + int(6*tot)

    s = set()
    len_triangular_C1s = len(triangular_C1s)
    i = 0
    while len(s) < len_triangular_C1s:
        val = a(i)
        s.update([val])
        i += 1

    ans2 = sorted(list(s))
    assert np.all(ans2[0:len(triangular_C1s)] == triangular_C1s)
    
    # square Ns
    # https://oeis.org/A001481
    # Quick and efficient
    # Translated from Mathematica
    # up to = 160; With[{max = Ceiling[Sqrt[upTo]]}, Select[Union[Total /@ (Tuples[Range[0, max], {2}]^2)], # <= upTo &]]  (* Harvey P. Dale, Apr 22 2011 *) 
    # 10 loops, best of 3: 17.3 ms per loop
    # Confirmed with SymPy
    up_to = 100000
    max_range = int(ceil(up_to**0.5))
    squares = [i*i for i in range(max_range+1)]
    seq = [i+j for i in squares for j in squares]
    seq = [i for i in set(seq) if i < up_to] # optional
    nums = seq[0:len(square_Ns)]
    assert_allclose(nums, square_Ns)

    # square C1s
    # https://oeis.org/A057961 is the sequence, there is one mathematica expression
    # but it needs SymPy or some hard work to be done
    # It is also the uniqiue elements in https://oeis.org/A057655
    # That has a convenient expression for pari, tested online and translated
    # a(n)=1+4*sum(k=0, sqrtint(n), sqrtint(n-k^2) ); /* Benoit Cloitre, Oct 08 2012 */ 
    # Currently 1.8 seconds
    # No numerical issues up to 35000 (confirmed with SymPy to do the root, int)
    def a2(n):
        sqrtint = lambda i: int(i**0.5)
        return 1 + 4*sum([sqrtint(n - k*k) for k in range(0, sqrtint(n) + 1)])
    
    ans = set([a2(i) for i in range(35000)])
    ans = sorted(list(ans))
    nums = ans[0:len(square_C1s)]
    assert_allclose(nums, square_C1s)



def test_Ntubes_HEDH():
    Ntubes_HEDH_c = [Ntubes_HEDH(DBundle=1.200-.008*2, Do=.028, pitch=.036, angle=i) for i in [30, 45, 60, 90]]
    assert_allclose(Ntubes_HEDH_c, [928, 804, 928, 804])
    
    with pytest.raises(Exception):
        # unsuported angle
        Ntubes_HEDH(DBundle=1.200-.008*2, Do=.028, pitch=.036, angle=20)
        
    with pytest.raises(Exception):
        # unsuported angle
        DBundle_for_Ntubes_HEDH(N=100, Do=.028, pitch=.036, angle=20)

    # Fuzzing test
    Do = 0.028
    for angle in [30, 45, 60, 90]:
        for pitch_ratio in [1.01, 1.1, 1.175, 1.25, 1.5, 1.75, 2]:
            pitch = Do*pitch_ratio
            for i in range(100):
                N = int(uniform(10, 10000))
                DBundle = DBundle_for_Ntubes_HEDH(N=N, Do=Do, pitch=pitch, angle=angle)
                # If we don't increase the bundle by a hair, int() will round the wrong way
                N_recalculated = Ntubes_HEDH(DBundle=DBundle*(1+1E-12), Do=Do, pitch=pitch, angle=angle)
                assert N == N_recalculated

def test_Ntubes_VDI():
    VDI_t = [[Ntubes_VDI(DBundle=1.184, Ntp=i, Do=.028, pitch=.036, angle=j) for i in [1,2,4,6,8]] for j in [30, 45, 60, 90]]
    VDI_values = [[983, 966, 929, 914, 903], [832, 818, 790, 778, 769], [983, 966, 929, 914, 903], [832, 818, 790, 778, 769]]
    assert_allclose(VDI_t, VDI_values)
    with pytest.raises(Exception):
        Ntubes_VDI(DBundle=1.184, Ntp=5, Do=.028, pitch=.036, angle=30)
    with pytest.raises(Exception):
        Ntubes_VDI(DBundle=1.184, Ntp=2, Do=.028, pitch=.036, angle=40)

    D_VDI =  [[D_for_Ntubes_VDI(N=970, Ntp=i, Do=0.00735, pitch=0.015, angle=j) for i in [1, 2, 4, 6, 8]] for j in [30, 60, 45, 90]]
    D_VDI_values = [[0.489981989464919, 0.5003600119829544, 0.522287673753684, 0.5311570964003711, 0.5377131635291736], [0.489981989464919, 0.5003600119829544, 0.522287673753684, 0.5311570964003711, 0.5377131635291736], [0.5326653264480428, 0.5422270203444146, 0.5625250342473964, 0.5707695340997739, 0.5768755899087357], [0.5326653264480428, 0.5422270203444146, 0.5625250342473964, 0.5707695340997739, 0.5768755899087357]]
    assert_allclose(D_VDI, D_VDI_values)

    with pytest.raises(Exception):
        D_for_Ntubes_VDI(N=970, Ntp=5., Do=0.00735, pitch=0.015, angle=30.)
    with pytest.raises(Exception):
        D_for_Ntubes_VDI(N=970, Ntp=2., Do=0.00735, pitch=0.015, angle=40.)


def test_Ntubes():
    methods = Ntubes(DBundle=1.2, Do=0.025, pitch=.025*1.25, AvailableMethods=True)
    Ntubes_calc = [Ntubes(DBundle=1.2, Do=0.025, pitch=.025*1.25, Method=i) for i in methods]
    assert Ntubes_calc == [1285, 1272, 1340, 1297]

    assert_allclose(Ntubes(DBundle=1.2, Do=0.025, pitch=.025*1.25), 1285)

    with pytest.raises(Exception):
        Ntubes(DBundle=1.2, Do=0.025, pitch=.025*1.25, Method='failure')

    D = size_bundle_from_tubecount(N=1285, Do=0.025, pitch=0.03125)
    assert_allclose(D, 1.1985676402390355)
    D = size_bundle_from_tubecount(N=1285, Do=0.025, pitch=0.03125, Method='HEDH')
    assert_allclose(D, 1.205810838411941)
    D = size_bundle_from_tubecount(N=1285, Do=0.025, pitch=0.03125, Method='VDI')
    assert_allclose(D, 1.1749025890472795)
    
    D = size_bundle_from_tubecount(N=13252, Do=.028, Ntp=2, angle=45, pitch=.028*1.25, Method='Perry')
    assert_allclose(D, 3.598336054740235)
    
    with pytest.raises(Exception):
        size_bundle_from_tubecount(N=1285, Do=0.025, pitch=0.03125, Method='BADMETHOD')

    l = size_bundle_from_tubecount(N=1285, Do=0.025, pitch=0.03125, AvailableMethods=True)
    assert len(l) == 4

def test_effectiveness_NTU():
    # Counterflow
    for i in range(20):
        eff = uniform(0, 1)
        Cr = uniform(0, 1)
        units = NTU_from_effectiveness(effectiveness=eff, Cr=Cr, subtype='counterflow')
        eff_calc = effectiveness_from_NTU(NTU=units, Cr=Cr, subtype='counterflow')
        assert_allclose(eff, eff_calc)
    # Case with Cr = 1
    NTU = NTU_from_effectiveness(effectiveness=.9, Cr=1, subtype='counterflow')
    assert_allclose(NTU, 9)
    e = effectiveness_from_NTU(NTU=9, Cr=1, subtype='counterflow')
    assert_allclose(e, 0.9)
        
        
    # Parallel
    for i in range(20):
        Cr = uniform(0, 1)
        eff = uniform(0, 1./(Cr + 1.)*(1-1E-7))
        units = NTU_from_effectiveness(effectiveness=eff, Cr=Cr, subtype='parallel')
        eff_calc = effectiveness_from_NTU(NTU=units, Cr=Cr, subtype='parallel')
        assert_allclose(eff, eff_calc)
        
    with pytest.raises(Exception):
        Cr = 0.6
        NTU_from_effectiveness(effectiveness=0.62500001, Cr=Cr, subtype='parallel')
        
        
    # Crossflow, Cmin mixed, Cmax unmixed
    
    for i in range(20):
        Cr = uniform(0, 1)
        eff = uniform(0, (1 - exp(-1/Cr))*(1-1E-7))
        N = NTU_from_effectiveness(eff, Cr=Cr, subtype='crossflow, mixed Cmin')
        eff_calc = effectiveness_from_NTU(N, Cr=Cr, subtype='crossflow, mixed Cmin')
        assert_allclose(eff, eff_calc)
        
    with pytest.raises(Exception):
        Cr = 0.7
        NTU_from_effectiveness(0.760348963559, Cr=Cr, subtype='crossflow, mixed Cmin')
        
            
    # Crossflow, Cmax mixed, Cmin unmixed
    for i in range(20):
        Cr = uniform(0, 1)
        eff = uniform(0, (exp(Cr) - 1)*exp(-Cr)/Cr-1E-5)
        N = NTU_from_effectiveness(eff, Cr=Cr, subtype='crossflow, mixed Cmax')
        eff_calc = effectiveness_from_NTU(N, Cr=Cr, subtype='crossflow, mixed Cmax')
        assert_allclose(eff, eff_calc)

    with pytest.raises(Exception):
        Cr = 0.7
        eff = 0.7201638517265581
        NTU_from_effectiveness(eff, Cr=Cr, subtype='crossflow, mixed Cmax')
        
    # Crossflow, this one needed a closed-form solver
    for i in range(100):
        Cr = uniform(0, 1)
        eff = uniform(0, 1)
        N = NTU_from_effectiveness(eff, Cr=Cr, subtype='crossflow approximate')
        eff_calc = effectiveness_from_NTU(N, Cr=Cr, subtype='crossflow approximate')
        assert_allclose(eff, eff_calc, rtol=1E-6) # brenth differs in old Python versions, rtol is needed 

    # Shell and tube - this one doesn't have a nice effectiveness limit,
    # and it depends on the number of shells
    
    for i in range(20):
        Cr = uniform(0, 1)
        shells = randint(1, 10)
        eff_max = (-((-Cr + sqrt(Cr**2 + 1) + 1)/(Cr + sqrt(Cr**2 + 1) - 1))**shells + 1)/(Cr - ((-Cr + sqrt(Cr**2 + 1) + 1)/(Cr + sqrt(Cr**2 + 1) - 1))**shells)
        eff = uniform(0, eff_max-1E-5)
        N = NTU_from_effectiveness(eff, Cr=Cr, subtype=str(shells)+'S&T')
        eff_calc = effectiveness_from_NTU(N, Cr=Cr, subtype=str(shells)+'S&T')
        assert_allclose(eff, eff_calc)
        
    with pytest.raises(Exception):
        NTU_from_effectiveness(.99, Cr=.7, subtype='5S&T')
        
    # Easy tests
    effectiveness = effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='crossflow, mixed Cmin')
    assert_allclose(effectiveness, 0.7497843941508544)
    NTU = NTU_from_effectiveness(effectiveness=effectiveness, Cr=0.7, subtype='crossflow, mixed Cmin')
    assert_allclose(NTU, 5)
    
    eff = effectiveness_from_NTU(NTU=5, Cr=0.7, subtype='crossflow, mixed Cmax')
    assert_allclose(eff, 0.7158099831204696)
    NTU = NTU_from_effectiveness(eff, Cr=0.7, subtype='crossflow, mixed Cmax')
    assert_allclose(5, NTU)
    
    eff = effectiveness_from_NTU(NTU=5, Cr=0, subtype='boiler')
    assert_allclose(eff, 0.9932620530009145)
    NTU = NTU_from_effectiveness(eff, Cr=0, subtype='boiler')
    assert_allclose(NTU, 5)
    
    with pytest.raises(Exception):
        effectiveness_from_NTU(NTU=5, Cr=1.01, subtype='crossflow, mixed Cmin')

    with pytest.raises(Exception):
        NTU_from_effectiveness(effectiveness=.2, Cr=1.01, subtype='crossflow, mixed Cmin')
        
        
    # bad names
    with pytest.raises(Exception):
        NTU_from_effectiveness(.99, Cr=.7, subtype='FAIL')
    with pytest.raises(Exception):
        effectiveness_from_NTU(NTU=5, Cr=.5, subtype='FAIL')


    # Crossflow analytical solution
    eff = effectiveness_from_NTU(NTU=5, Cr=.7, subtype='crossflow')
    assert_allclose(eff, 0.8444821799748551)

    def crossflow_unmixed_sum_infinite(NTU, Cr):
        def Pn(NTU, n):
            tot = sum([(n+1.-j)/factorial(j)*NTU**(n+j) for j in range(1, n+1)])
            return tot/factorial(n+1.)
        tot = sum([Cr**n*Pn(NTU, n) for n in range(1, 150)])
        return 1 - exp(-NTU) - exp(-(1+Cr)*NTU)*tot
    
    eff_old = crossflow_unmixed_sum_infinite(5, .7)
    assert_allclose(eff, eff_old)

    # Crossflow analytical, this one needed a closed-form solver
    for i in range(20):
        Cr = uniform(0, 1)
        eff = uniform(0, .9)
        # A good anser is not always obtainable at eff> 0.9 at very high NTU,
        # because the integral term gets too close to 1 for floating point numbers
        # to capture any more accuracy
        # This is not likely to be a problem to users
        N = NTU_from_effectiveness(eff, Cr=Cr, subtype='crossflow')
        eff_calc = effectiveness_from_NTU(N, Cr=Cr, subtype='crossflow')
        assert_allclose(eff, eff_calc, rtol=1E-6) # brenth differs in old Python versions, rtol is needed 
    
    
    
def test_effectiveness_NTU_method():
    ans_known = {'Q': 192850.0, 'Thi': 130, 'Cmax': 9672.0, 'Tho': 110.06100082712986, 'Cmin': 2755.0, 'NTU': 1.1040839095588, 'Tco': 85, 'Tci': 15, 'Cr': 0.2848428453267163, 'effectiveness': 0.6086956521739131, 'UA': 3041.751170834494}
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Tho=110.06100082712986)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Thi=130)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]

    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Thi=130, Tho=110.06100082712986, Tci=15)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Thi=130, Tho=110.06100082712986, Tco=85)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]

    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tco=85, Tho=110.06100082712986, UA=3041.751170834494)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Thi=130, UA=3041.751170834494)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tho=110.06100082712986, UA=3041.751170834494)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tco=85, Thi=130, UA=3041.751170834494)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Tho=110.06100082712986, UA=3041.751170834494)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tco=85, Thi=130, Tho=110.06100082712986, UA=3041.751170834494)
    [assert_allclose(ans_known[i], ans[i]) for i in ans_known.keys()]

    with pytest.raises(Exception):
        # Test raising an error with only on set of stream information 
        effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Thi=130, Tho=110.06100082712986, UA=3041.751170834494)
        
    with pytest.raises(Exception):
        # Inconsistent hot and cold temperatures and heat capacity ratios
        effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Thi=130, Tho=110.06100082712986, Tco=85, Tci=5)

    with pytest.raises(Exception):
        # Calculate UA, but no code side temperature information given
        effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Thi=130, Tho=110.06100082712986)
        
    with pytest.raises(Exception):
        # Calculate UA, but no hot side temperature information given
        effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tco=85)
        
    with pytest.raises(Exception):
        # Calculate UA, but only two temperatures given
        effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Thi=130)
        
def test_F_LMTD_Fakheri():
    '''Number of tube passes must be a multiple of 2N for correlation to work.
    N can be 1.

    Example from http://excelcalculations.blogspot.ca/2011/06/lmtd-correction-factor.html 
    spreadsheet file which Bowman et al (1940).
    This matches for 3, 6, and 11 shell passes perfectly.

    This also matches that from the sheet: 
    http://www.mhprofessional.com/getpage.php?c=0071624082_download.php&cat=113
    '''    
    F_calc = F_LMTD_Fakheri(Tci=15, Tco=85, Thi=130, Tho=110, shells=1)
    assert_allclose(F_calc, 0.9438358829645933)
    
    # R = 1 check
    F_calc = F_LMTD_Fakheri(Tci=15, Tco=35, Thi=130, Tho=110, shells=1)
    assert_allclose(F_calc, 0.9925689447100824)
    
    for i in range(1, 10):
        ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype=str(i)+'S&T', Tci=15, Tco=85, Thi=130)
        dTlm = LMTD(Thi=130, Tho=110.06100082712986,  Tci=15, Tco=85)
        F_expect = ans['Q']/ans['UA']/dTlm
        
        F_calc = F_LMTD_Fakheri(Tci=15, Tco=85, Thi=130, Tho=110.06100082712986, shells=i)
        assert_allclose(F_expect, F_calc)
        F_calc = F_LMTD_Fakheri(Thi=15, Tho=85, Tci=130, Tco=110.06100082712986, shells=i)
        assert_allclose(F_expect, F_calc)


def test_temperature_effectiveness_basic():
    # Except for the crossflow mixed 1&2 cases, taken from an example and checked that
    # it matches the e-NTU method. The approximate formula for crossflow is somewhat
    # different - it is believed the approximations are different.
    
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='counterflow')
    assert_allclose(P1, 0.173382601503)
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='parallel')
    assert_allclose(P1, 0.163852912049)
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='crossflow approximate')
    assert_allclose(P1, 0.149974594007)
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='crossflow')
    assert_allclose(P1, 0.1698702121873175)
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='crossflow, mixed 1')
    assert_allclose(P1, 0.168678230894)
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='crossflow, mixed 2')
    assert_allclose(P1, 0.16953790774)
    P1 = temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='crossflow, mixed 1&2')
    assert_allclose(P1, 0.168411216829)

    with pytest.raises(Exception):
        temperature_effectiveness_basic(R1=3.5107078039927404, NTU1=0.29786672449248663, subtype='FAIL')
        
    # Formulas are in [1]_, [3]_, and [2]_.
    
def test_temperature_effectiveness_TEMA_J():
    # All three models are checked with Rosenhow and then Shaw
    # Formulas presented in Thulukkanam are with respect to the other side of the
    # exchanger
    P1 = temperature_effectiveness_TEMA_J(R1=1/3., NTU1=1., Ntp=1)
    assert_allclose(P1, 0.5699085193651295)
    P1 = temperature_effectiveness_TEMA_J(R1=2., NTU1=1.,  Ntp=1) # R = 2 case
    assert_allclose(P1, 0.3580830895954234)
    P1 = temperature_effectiveness_TEMA_J(R1=1/3., NTU1=1., Ntp=2)
    assert_allclose(P1, 0.5688878232315694)
    P1 = temperature_effectiveness_TEMA_J(R1=1/3., NTU1=1., Ntp=4)
    assert_allclose(P1, 0.5688711846568247)
    
    with pytest.raises(Exception):
        temperature_effectiveness_TEMA_J(R1=1/3., NTU1=1., Ntp=3)
    
    

        
      
def test_temperature_effectiveness_TEMA_H():
    P1 = temperature_effectiveness_TEMA_H(R1=1/3., NTU1=1., Ntp=1)
    assert_allclose(P1, 0.5730728284905833)
    P1 = temperature_effectiveness_TEMA_H(R1=2., NTU1=1., Ntp=1) # R = 2 case
    assert_allclose(P1, 0.3640257049950876)
    P1 = temperature_effectiveness_TEMA_H(R1=1/3., NTU1=1., Ntp=2)
    assert_allclose(P1, 0.5824437803128222)
    P1 = temperature_effectiveness_TEMA_H(R1=4., NTU1=1., Ntp=2) # R = 4 case
    assert_allclose(P1, 0.2366953352462191)
    
    P1 = temperature_effectiveness_TEMA_H(R1=1/3., NTU1=1., Ntp=2, optimal=False)
    assert_allclose(P1, 0.5560057072310012)
    P1 = temperature_effectiveness_TEMA_H(R1=4, NTU1=1., Ntp=2, optimal=False)
    assert_allclose(P1, 0.19223481412807347) # R2 = 0.25
    
    # The 1 and 2 case by default are checked with Rosenhow and Shah
    # for the two pass unoptimal case, the result is from Thulukkanam only.
    # The 2-pass optimal arrangement from  Rosenhow and Shaw is the same
    # as that of Thulukkanam however, and shown below.
    m1 = .5
    m2 = 1.2
    Cp1 = 1800.
    Cp2 = 2200.
    UA = 500.
    C1 = m1*Cp1
    C2 = m2*Cp2
    R1_orig = R1 = C1/C2
    NTU1 = UA/C1
    R2 = C2/C1
    NTU2 = UA/C2
    
    R1 = R2
    NTU1 = NTU2
    
    alpha = NTU1*(4*R1 + 1)/8.
    beta = NTU1*(4*R1 - 1)/8.
    D = (1 - exp(-alpha))/(4.*R1 + 1)
    
    E = (1 - exp(-beta))/(4*R1 - 1)
    H = (1 - exp(-2*beta))/(4.*R1 - 1)
    
    G = (1-D)**2*(D**2 + E**2) + D**2*(1+E)**2
    B = (1 + H)*(1 + E)**2
    P1 = (1 - (1-D)**4/(B - 4.*R1*G))
    P1 = P1/R1_orig 
    assert_allclose(P1, 0.40026600037802335)


    with pytest.raises(Exception):
        temperature_effectiveness_TEMA_H(R1=1/3., NTU1=1., Ntp=5)


def test_temperature_effectiveness_TEMA_G():
        # Checked with Shah and Rosenhow, formula typed and then the other case is working
    P1 = temperature_effectiveness_TEMA_G(R1=1/3., NTU1=1., Ntp=1)
    assert_allclose(P1, 0.5730149350867675)
    P1 = temperature_effectiveness_TEMA_G(R1=1/3., NTU1=1., Ntp=2) # TEST CASSE
    assert_allclose(P1, 0.5824238778134628)
    
    # Ntp = 1, R=1 case
    P1_Ntp_R1 = 0.8024466201983814
    P1 = temperature_effectiveness_TEMA_G(R1=1., NTU1=7., Ntp=1) # R = 1 case
    assert_allclose(P1, P1_Ntp_R1)
    P1_near = temperature_effectiveness_TEMA_G(R1=1-1E-9, NTU1=7, Ntp=1)
    assert_allclose(P1_near, P1_Ntp_R1)
    
    # Ntp = 2, optimal, R=2 case
    P1_Ntp_R1 = 0.4838424889135673
    P1 = temperature_effectiveness_TEMA_G(R1=2., NTU1=7., Ntp=2) # R = 2 case
    assert_allclose(P1, P1_Ntp_R1)
    P1_near = temperature_effectiveness_TEMA_G(R1=2-1E-9, NTU1=7., Ntp=2)
    assert_allclose(P1_near, P1_Ntp_R1)


    # Ntp = 2, not optimal case, R1=0.5 case
    P1 = temperature_effectiveness_TEMA_G(R1=1/3., NTU1=1., Ntp=2, optimal=False)
    assert_allclose(P1, 0.5559883028569507)

    P1_Ntp_R1 = 0.3182960796403764
    P1 = temperature_effectiveness_TEMA_G(R1=2, NTU1=1., Ntp=2, optimal=False)
    assert_allclose(P1, P1_Ntp_R1)
    P1_near = temperature_effectiveness_TEMA_G(R1=2-1E-9, NTU1=1., Ntp=2, optimal=False)
    assert_allclose(P1_near, P1_Ntp_R1)
    
    with pytest.raises(Exception):
        temperature_effectiveness_TEMA_G(R1=2., NTU1=7., Ntp=5)

    # The optimal 2 pass case from Thulukkanam is checked with the following case
    # to be the same as those in Rosenhow and Shah
    # Believed working great.
    m1 = .5
    m2 = 1.2
    Cp1 = 1800.
    Cp2 = 2200.
    UA = 500.
    C1 = m1*Cp1
    C2 = m2*Cp2
    R1_orig = R1 = C1/C2
    NTU1 = UA/C1
    R2 = C2/C1
    NTU2 = UA/C2
    
    P1_good = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1, Ntp=2)


    # Good G 2 pass case, working
    R1 = R2
    NTU1 = NTU2
    
    beta = exp(-NTU1*(2*R1 - 1)/2.)
    alpha = exp(-NTU1*(2*R1 + 1)/4.)
    B = (4*R1 - beta*(2*R1 + 1))/(2*R1 - 1.)
    A = -1*(1-alpha)**2/(R1 + 0.5)
    P1 = (B - alpha**2)/(R1*(A + 2 + B/R1))
    
    P1 = P1/R1_orig 
    assert_allclose(P1, P1_good)
    
    
    
def test_temperature_effectiveness_TEMA_E():
    # 1, 2 both cases are perfect
    eff = temperature_effectiveness_TEMA_E(R1=1/3., NTU1=1., Ntp=1)
    assert_allclose(eff, 0.5870500654031314)
    eff = temperature_effectiveness_TEMA_E(R1=1., NTU1=7., Ntp=1)
    assert_allclose(eff, 0.875)
    
    # Remaining E-shells, checked
    eff = temperature_effectiveness_TEMA_E(R1=1/3., NTU1=1., Ntp=2)
    assert_allclose(eff, 0.5689613217664634)
    eff = temperature_effectiveness_TEMA_E(R1=1., NTU1=7., Ntp=2) # R = 1 case
    assert_allclose(eff, 0.5857620762776082)
    
    eff = temperature_effectiveness_TEMA_E(R1=1/3., NTU1=1., Ntp=2, optimal=False)
    assert_allclose(eff, 0.5699085193651295) # unoptimal case
    eff = temperature_effectiveness_TEMA_E(R1=2, NTU1=1., Ntp=2, optimal=False)
    assert_allclose(eff, 0.3580830895954234)
    
    
    
    eff = temperature_effectiveness_TEMA_E(R1=1/3., NTU1=1., Ntp=3)
    assert_allclose(eff, 0.5708624888990603)
    eff = temperature_effectiveness_TEMA_E(R1=1., NTU1=7., Ntp=3) # R = 1 case
    assert_allclose(eff, 0.6366132064792461)
    
    eff = temperature_effectiveness_TEMA_E(R1=3., NTU1=1., Ntp=3, optimal=False)
    assert_allclose(eff, 0.276815590660033)
    
    eff = temperature_effectiveness_TEMA_E(R1=1/3., NTU1=1., Ntp=4)
    assert_allclose(eff, 0.56888933865756)
    eff = temperature_effectiveness_TEMA_E(R1=1., NTU1=7., Ntp=4) # R = 1 case, even though it's no longer used
    assert_allclose(eff, 0.5571628802075902)
    
    
    with pytest.raises(Exception):
         temperature_effectiveness_TEMA_E(R1=1., NTU1=7., Ntp=7)

    # Compare the expression for 4 tube passes in two of the sources with that
    # in the third.
    R1 = 1/3.
    NTU1 = 1
    D = (4 + R1**2)**0.5
    B = tanh(R1*NTU1/4.)
    A = 1/tanh(D*NTU1/4.)
    P1 = 4*(2*(1 + R1) + D*A + R1*B)**-1
    assert_allclose(P1, 0.56888933865756)
    
    
def test_temperature_effectiveness_air_cooler():
    # 1 pass-N rows case
    R1 = 0.9090909090909091
    NTU1 = 14.958251192851375
    
    expected_P1s = [0.6568205178185993, 0.7589599992302802, 0.8064227529035781, 0.8330202134563712, 0.8491213831157698, 0.8594126317585193, 0.8662974164766494, 0.871087594489211, 0.8745345926002213, 0.8770877118478316, 0.8790262425246239, 0.8805299599498708, 0.8817182454510963, 0.8826726050451953, 0.8834500769975893, 0.8840914654885264, 0.8846265414931143, 0.88507741320138, 0.8854607616314836, 0.8857893552314147, 0.886073095973165, 0.8863197546874396, 0.8865354963468465, 0.8867252608860744, 0.8868930430686396]
    P1s_calc = [temperature_effectiveness_air_cooler(R1=R1, NTU1=NTU1, rows=N, passes=1) for N in  range(1, 26)]
    assert_allclose(expected_P1s, P1s_calc)
    
    # Compare the results of 1-N against the function without the annoying optimizations;
    # may be helpful for debugging
    def calc_N_1_orig(NTU1, R1, N):
        NTU, R = NTU1, R1
        K = 1 - exp(-NTU/N)
        top = N*exp(N*K*R)
    
        tot = 0
        for i in range(1, N):
            for j in range(0, i+1):
                prod = factorial(i)/factorial(i-j)/factorial(j)
                tot1 = prod*K**j*exp(-(i-j)*NTU/N)
                tot2 = 0
                for k in range(0, j+1):
                    tot2 += (N*K*R)**k/factorial(k)
    
                tot += tot1*tot2
    
        P = 1/R*(1 - (top/(1+tot))**-1)
        return P
    P1s_calc = [calc_N_1_orig(R1=R1, NTU1=NTU1, N=N) for N in  range(1, 26)]
    assert_allclose(expected_P1s, P1s_calc)
    
    
    # N rows / N passes (N from 2 to 5) cases
    R1, NTU1 = 1.1, .5
    expected_P1s = [0.3254086785640332, 0.3267486216405819, 0.3272282999575143, 0.3274325680785421]
    P1s_calc = [temperature_effectiveness_air_cooler(R1, NTU1, rows=N, passes=N) for N in  range(2, 6)]
    assert_allclose(expected_P1s, P1s_calc)
    
    # 4 row / 2 pass special case
    P1_calc = temperature_effectiveness_air_cooler(R1, NTU1, rows=4, passes=2)
    assert_allclose(P1_calc, 0.32552127419957044)
    
    # Tentative checking of the above has been done with hete.c for isolated cases
    
def test_temperature_effectiveness_air_cooler_coerce():
    # Simple test a call that the number of row and passes can be domain reduced
    # without causing a recursion depth error
    # Do not test the results for any specific answer, as they will ideally one day
    # be replaced with the exactly correct one
    [temperature_effectiveness_air_cooler(.5, 2, rows=j, passes=i) for i in range(1, 10) for j in range(1, 10)]

    
@pytest.mark.mpmath
def test_P_NTU_method():
    # Counterflow case
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='counterflow', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1i=130, T2i=15, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    # Parallel flow case
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='parallel', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1i=130, T2i=15, subtype='parallel')
    assert_allclose(ans2['Q'], ans['Q'])
    # Mixed Cmax/ 1
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1i=130, T2i=15, subtype='crossflow, mixed 1')
    assert_allclose(ans2['Q'], ans['Q'])
    # Mixed Cmin/2
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmin', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1i=130, T2i=15, subtype='crossflow, mixed 2')
    assert_allclose(ans2['Q'], ans['Q'])
    
    # Counterflow case but with all five different temperature input cases (both inlets known already done)
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='counterflow', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1o=110.06100082712986, T2o=85, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1i=130, T2o=85, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1o=110.06100082712986, T2i=15, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T2o=85, T2i=15, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=ans['UA'], T1o=110.06100082712986, T1i=130, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    
    # Only 1 temperature input
    with pytest.raises(Exception):
        P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, subtype='counterflow')
        
    # Bad HX type input
    with pytest.raises(Exception):
        P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='BADTYPE')

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='E', Ntp=10)
    assert_allclose(ans['Q'], 32212.185563086336,)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='G', Ntp=2)
    assert_allclose(ans['Q'], 32224.88788570008)
    
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='H', Ntp=2)
    assert_allclose(ans['Q'], 32224.888572366734)
    
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='J', Ntp=2)
    assert_allclose(ans['Q'], 32212.185699719837)
    
    # Plate tests
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='3/1')
    assert_allclose(ans['Q'], 32214.179745602625)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='3/1', optimal=False)
    assert_allclose(ans['Q'], 32210.4190840378)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='2/2')
    assert_allclose(ans['Q'], 32229.120739501937)
    
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='2/2', optimal=False)
    assert_allclose(ans['Q'], 32203.721238671216)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='2/2c', optimal=False)
    assert_allclose(ans['Q'], 32203.721238671216)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., UA=300, T1i=130, T2i=15, subtype='2/2p', optimal=False)
    assert_allclose(ans['Q'], 32195.273806845064)


def test_P_NTU_method_backwards():
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='counterflow', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T2i=15, T2o=85, T1o=110.06100082712986, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
#    # Parallel flow case
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='parallel', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1i=130, T2i=15, T1o=110.06100082712986, subtype='parallel')
    assert_allclose(ans2['Q'], ans['Q'])
#    # Mixed Cmax/ 1
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmax', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T1i=130, T2i=15, subtype='crossflow, mixed 1')
    assert_allclose(ans2['Q'], ans['Q'])
#    # Mixed Cmin/2
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='crossflow, mixed Cmin', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T1i=130, T2i=15, subtype='crossflow, mixed 2')
    assert_allclose(ans2['Q'], ans['Q'])
    
#    # Counterflow case but with all five different temperature input cases (both inlets known already done)
    ans = effectiveness_NTU_method(mh=5.2, mc=1.45, Cph=1860., Cpc=1900, subtype='counterflow', Tci=15, Tco=85, Tho=110.06100082712986)
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1i=130, T1o=110.06100082712986, T2o=85, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T1i=130, T2o=85, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1i=130, T1o=110.06100082712986, T2i=15, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T2o=85, T2i=15, T1o=110.06100082712986, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T1i=130, T2i=15, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])


    ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T2o=85, T1i=130, T2i=15, subtype='counterflow')
    assert_allclose(ans2['Q'], ans['Q'])
    
    # TEMA types
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1i=130, T1o=126.66954243557834, T2i=15, subtype='E', Ntp=10)
    assert_allclose(ans['Q'], 32212.185563086336,)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.66822912678866, T1i=130, T2i=15, subtype='G', Ntp=2)
    assert_allclose(ans['Q'], 32224.88788570008)
    
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.66822905579335, T1i=130, T2i=15, subtype='H', Ntp=2)
    assert_allclose(ans['Q'], 32224.888572366734)
    
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.66954242145162, T1i=130, T2i=15, subtype='J', Ntp=2)
    assert_allclose(ans['Q'], 32212.185699719837)
    
    # Plate tests
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.6693362545903, T1i=130, T2i=15, subtype='3/1')
    assert_allclose(ans['Q'], 32214.179745602625)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.66972507402421, T1i=130, T2i=15, subtype='3/1', optimal=False)
    assert_allclose(ans['Q'], 32210.4190840378)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.66779148681742, T1i=130, T2i=15, subtype='2/2')
    assert_allclose(ans['Q'], 32229.120739501937)
  
    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.67041757251124, T1i=130, T2i=15, subtype='2/2', optimal=False)
    assert_allclose(ans['Q'], 32203.721238671216)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.67041757251124, T1i=130, T2i=15, subtype='2/2c', optimal=False)
    assert_allclose(ans['Q'], 32203.721238671216)

    ans = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=126.67129096289857, T1i=130, T2i=15, subtype='2/2p', optimal=False)
    assert_allclose(ans['Q'], 32195.273806845064)

    
    
    
    
    
    
    # Q for both streams don't match case
    with pytest.raises(Exception):
        P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T2o=85, T1i=170, T2i=15, subtype='counterflow')
    # No T speced on side 2
    with pytest.raises(Exception):
        P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T1i=130, subtype='counterflow')
    # No T specified on side 1
    with pytest.raises(Exception):
        P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T2o=85, T2i=15, subtype='counterflow')
    # No T information at all
    with pytest.raises(Exception):
        ans2 = P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., subtype='counterflow')
    # subtype not recognized
    with pytest.raises(Exception):
        P_NTU_method(m1=5.2, m2=1.45, Cp1=1860., Cp2=1900., T1o=110.06100082712986, T1i=130, T2i=15, subtype='NOTAREALTYPEOFHEATEXCHANGER')




        


def test_Pp():
    from ht.hx import Pp, Pc
    # randomly chosen test value
    ans = Pp(5, .4)
    assert_allclose(ans, 0.713634370024604)
    
    # Test the limit works with a small difference
    assert_allclose(Pp(2, -1), Pp(2, -1+1E-9))
    
    # randomly chosen test value
    assert_allclose(Pc(5, .7), 0.9206703686051108)
    # Test the limit works with a small difference
    assert_allclose(Pc(5, 1), Pc(5, 1-1E-8))


def test_temperature_effectiveness_plate():
    R1 = 0.5
    NTU1 = 1.5
    
    P1 = temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=1, counterflow=True)
    assert_allclose(P1, 0.6907854082479168)
    P1 = temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=1, counterflow=False)
    assert_allclose(P1, 0.5964005169587571)
    
    # 1 pass/2 pass
    for b1 in [True, False]:
        for b2 in [True, False]:
            P1 = temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=2, counterflow=b1, passes_counterflow=b2)
            assert_allclose(P1, 0.6439306988115887)
            # We can check we did the conversion right as follows:
            NTU2 = NTU1*R1 #
            R2 = 1./R1 # switch 2
            P2 = P1*R1
            P2_reversed = temperature_effectiveness_plate(R2, NTU2, Np1=2, Np2=1) 
            assert_allclose(P2, P2_reversed)

            # in reverse
            P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=1, counterflow=b1, passes_counterflow=b2) 
            assert_allclose(P1, 0.6505342399575915)
            


    
    # 1 pass/3 pass, counterflow
    for b1 in [True, False]:
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=3, counterflow=True, passes_counterflow=b1)
        assert_allclose(P1, 0.6491132138517642)
        # In reverse
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=3, Np2=1, counterflow=True, passes_counterflow=b1)
        assert_allclose(P1, 0.6565261377239298)
    
    # 1 pass/3 pass, parallel
    for b1 in [True, False]:
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=3, counterflow=False, passes_counterflow=b1)
        assert_allclose(P1, 0.6385443460862099)
        # in reverse
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=3, Np2=1, counterflow=False, passes_counterflow=b1)
        assert_allclose(P1, 0.6459675147406085)
    
    # 1 pass/4 pass
    for b1 in [True, False]:
        for b2 in [True, False]:
            P1 = temperature_effectiveness_plate(R1, NTU1, Np1=1, Np2=4, counterflow=b1, passes_counterflow=b2)
            assert_allclose(P1, 0.6438068496552443)
            # In reverse
            P1 = temperature_effectiveness_plate(R1, NTU1, Np1=4, Np2=1, counterflow=b1, passes_counterflow=b2)
            assert_allclose(P1, 0.6515539888566283)
            
            
            
    # Four different results for 4 passes
    P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=2, counterflow=False, passes_counterflow=False)
    assert_allclose(P1, 0.5964005169587571)
    P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=2, counterflow=False, passes_counterflow=True)
    assert_allclose(P1, 0.6123845839665905)
    P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=2, counterflow=True, passes_counterflow=False)
    assert_allclose(P1, 0.6636659009073801)
    P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=2, counterflow=True, passes_counterflow=True)
    assert_allclose(P1, 0.6907854082479168)
    
    # 2-3 counterflow
    for b1 in [True, False]:
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=3, counterflow=True, passes_counterflow=b1)
        assert_allclose(P1, 0.67478876724034)
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=3, counterflow=False, passes_counterflow=b1)
        assert_allclose(P1, 0.6102922060616937)
        # In reverse
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=3, Np2=2, counterflow=True, passes_counterflow=b1)
        assert_allclose(P1, 0.675522913050678)
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=3, Np2=2, counterflow=False, passes_counterflow=b1)
        assert_allclose(P1, 0.6105764872072659)
    
    
    # 2-4 counterflow
    for b1 in [True, False]:
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=4, counterflow=True, passes_counterflow=b1)
        assert_allclose(P1, 0.6777107269336475)
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=2, Np2=4, counterflow=False, passes_counterflow=b1)
        assert_allclose(P1, 0.6048585344522575)
        # In reverse
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=4, Np2=2, counterflow=True, passes_counterflow=b1)
        assert_allclose(P1, 0.6786601861219819)
        P1 = temperature_effectiveness_plate(R1, NTU1, Np1=4, Np2=2, counterflow=False, passes_counterflow=b1)
        assert_allclose(P1, 0.6054166111196166)


    with pytest.raises(Exception):
        temperature_effectiveness_plate(R1=1/3., NTU1=1., Np1=3, Np2=3)
        
    
@pytest.mark.mpmath
def test_NTU_from_P_basic():
    # Analytical result for counterflow
    R1s = np.logspace(np.log10(2E-5), np.log10(1E2), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(1E2), 10000)
    
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            # Not all of the guesses work forward; some overflow, some divide by 0
            P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='counterflow')
            # Backwards, it's the same divide by zero or log(negative number)
            NTU1_calc = NTU_from_P_basic(P1, R1, subtype='counterflow')
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        # Again, multiple values of NTU1 can produce the same P1
        P1_calc = temperature_effectiveness_basic(R1=R1, NTU1=NTU1_calc, subtype='counterflow')
        assert_allclose(P1, P1_calc)
        
    # Analytical result for parallel flow
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='parallel')
            # Backwards, it's the same divide by zero or log(negative number)
            NTU1_calc = NTU_from_P_basic(P1, R1, subtype='parallel')
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        P1_calc = temperature_effectiveness_basic(R1=R1, NTU1=NTU1_calc, subtype='parallel')
        assert_allclose(P1, P1_calc)

    # Analytical result for 'crossflow, mixed 1'
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            # Not all of the guesses work forward; some overflow, some divide by 0
            P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow, mixed 1')
            # Backwards, it's the same divide by zero or log(negative number)
            NTU1_calc = NTU_from_P_basic(P1, R1, subtype='crossflow, mixed 1')
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        # Again, multiple values of NTU1 can produce the same P1
        P1_calc = temperature_effectiveness_basic(R1=R1, NTU1=NTU1_calc, subtype='crossflow, mixed 1')
        assert_allclose(P1, P1_calc)

    # Analytical result for 'crossflow, mixed 2'
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            # Not all of the guesses work forward; some overflow, some divide by 0
            P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow, mixed 2')
            # Backwards, it's the same divide by zero or log(negative number)
            NTU1_calc = NTU_from_P_basic(P1, R1, subtype='crossflow, mixed 2')
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        # Again, multiple values of NTU1 can produce the same P1
        P1_calc = temperature_effectiveness_basic(R1=R1, NTU1=NTU1_calc, subtype='crossflow, mixed 2')
        assert_allclose(P1, P1_calc)

    
    # Test 'crossflow, mixed 1&2':
    R1s = np.logspace(np.log10(2E-5), np.log10(1E2), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(1E2), 10000)

    seed(0)
    tot = 0
    for i in range(100):
        R1 = choice(R1s)
        NTU1 = choice(NTU1s)
        
        P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow, mixed 1&2')
        try:
            # Very rarely, the pade approximation will get a result too close to the infeasibility region and
            # the solver cannot start as it is already outside the region
            NTU1_calc = NTU_from_P_basic(P1, R1, subtype='crossflow, mixed 1&2')
        except:
            continue
        # May not get the original NTU1, but the found NTU1 needs to produce the same P1.
        P1_calc = temperature_effectiveness_basic(R1=R1, NTU1=NTU1_calc, subtype='crossflow, mixed 1&2')
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot == 100


    # crossflow approximate - easy as 1 is always a possibility for any R
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow approximate')
        NTU1_calc = NTU_from_P_basic(P1, R1, subtype='crossflow approximate')
        # We have to compare the re calculated P1 values, because for many values of NTU1,
        # at the initial far guess of 10000 P1 = 1 and at the random NTU1 P1 is also 1
        P1_calc = temperature_effectiveness_basic(R1=R1, NTU1=NTU1_calc, subtype='crossflow approximate')
        # In python 2.6 and 3.3 the solver doesn't converge as well, so we need
        # to add a little tolerance
        assert_allclose(P1, P1_calc, rtol=5E-6)
        
    # Crossflow approximate test case
    R1 = .1
    NTU1 = 2
    P1_calc_orig = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow approximate')
    P1_expect = 0.8408180737140558
    assert_allclose(P1_calc_orig, P1_expect)
    NTU1_backwards = NTU_from_P_basic(P1=P1_expect, R1=R1, subtype='crossflow approximate')
    assert_allclose(NTU1, NTU1_backwards)
        
        
    # Test cross flow - failes VERY OFTEN, should rely on crossflow approximate
    NTU1 = 10
    R1 = 0.5
    P1 = temperature_effectiveness_basic(R1=R1, NTU1=NTU1, subtype='crossflow')
    NTU1_calc = NTU_from_P_basic(P1, R1=R1, subtype='crossflow')
    assert_allclose(NTU1, NTU1_calc)
    
    # bad type of exchanger
    with pytest.raises(Exception):
        NTU_from_P_basic(P1=.975, R1=.1, subtype='BADTYPE')


@pytest.mark.mpmath
def test_NTU_from_P_E():
    # not yet documented
    
    # 1 tube pass AKA counterflow
    R1s = np.logspace(np.log10(2E-5), np.log10(1E2), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(1E2), 10000)
    
    # Exact same asa as the counterflow basic case
    tot = 0
    seed(0)
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            # Not all of the guesses work forward; some overflow, some divide by 0
            P1 = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1, Ntp=1)
            # Backwards, it's the same divide by zero or log(negative number)
            NTU1_calc = NTU_from_P_E(P1, R1, Ntp=1)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        # Again, multiple values of NTU1 can produce the same P1
        P1_calc = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1_calc, Ntp=1)
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot >= 85
    
    # 2 tube passes (optimal arrangement) (analytical)
    R1 = 1.1
    NTU1 = 10
    P1 = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1, Ntp=2, optimal=True)
    P1_expect = 0.5576299522073297
    assert_allclose(P1, P1_expect)
    NTU1_calc = NTU_from_P_E(P1, R1, Ntp=2, optimal=True)
    assert_allclose(NTU1_calc, NTU1)

    # 2 tube pass (unoptimal)
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            # Not all of the guesses work forward; some overflow, some divide by 0
            P1 = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1, Ntp=2, optimal=False)
            # Backwards, it's the same divide by zero or log(negative number)
            NTU1_calc = NTU_from_P_E(P1, R1, Ntp=2, optimal=False)
        except (ValueError, OverflowError, ZeroDivisionError):
            continue
        # Again, multiple values of NTU1 can produce the same P1
        try:
            P1_calc = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1_calc, Ntp=2, optimal=False)
        except (ZeroDivisionError):
            continue
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot >= 90
    
    # At the default mpmath precision, the following will predict a value larger
    # than one
    bad_P1 = temperature_effectiveness_TEMA_E(R1=1E-8 , NTU1=19.60414246043446, Ntp=2, optimal=False)
    assert_allclose(bad_P1, 1.0000000050247593)

    # 4 pass
    for Ntp in [4, 6, 8, 10, 12]:
        tot = 0
        for i in range(100):
            R1 = float(choice(R1s))
            NTU1 = float(choice(NTU1s))
            try:
                P1 = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1, Ntp=Ntp)
                NTU1_calc = NTU_from_P_E(P1, R1, Ntp=Ntp)
            except ValueError:
                # The case where with mpmath being used, the result is too high for
                # the bounded solver to be able to solve it
                continue
            P1_calc = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1_calc, Ntp=Ntp)
            assert_allclose(P1, P1_calc)
            tot +=1
        assert tot >= 70

    # 3 pass optimal and not optimal
    R1s = np.logspace(np.log10(2E-5), np.log10(1E1), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(1E1), 10000)
    
    seed(0)
    for optimal in [True, False]:
        tot = 0
        for i in range(100):
            R1 = float(choice(R1s))
            NTU1 = float(choice(NTU1s))
            try:
                P1 = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1, Ntp=3, optimal=optimal)
                NTU1_calc = NTU_from_P_E(P1, R1, Ntp=3, optimal=optimal)
            except (ValueError):
                # The case where with mpmath being used, the result is too high for
                # the bounded solver to be able to solve it
                continue
            # Again, multiple values of NTU1 can produce the same P1
            P1_calc = temperature_effectiveness_TEMA_E(R1=R1, NTU1=NTU1_calc, Ntp=3, optimal=optimal)
            assert_allclose(P1, P1_calc)
            tot +=1
        assert tot >= 97

    with pytest.raises(Exception):
        NTU_from_P_E(P1=1, R1=1, Ntp=17)


@pytest.mark.mpmath
def test_NTU_from_P_H():
    # Within these limits everything is fund
    R1s = np.logspace(np.log10(2E-5), np.log10(1E1), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(10), 10000)
    
    seed(0)
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        P1 = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1, Ntp=1)
        NTU1_calc = NTU_from_P_H(P1, R1, Ntp=1)
        P1_calc = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1_calc, Ntp=1)
        assert_allclose(P1, P1_calc)
        
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        P1 = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1, Ntp=2)
        NTU1_calc = NTU_from_P_H(P1, R1, Ntp=2)
        P1_calc = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1_calc, Ntp=2)
        assert_allclose(P1, P1_calc)

    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        P1 = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1, Ntp=2, optimal=False)
        NTU1_calc = NTU_from_P_H(P1, R1, Ntp=2, optimal=False)
        P1_calc = temperature_effectiveness_TEMA_H(R1=R1, NTU1=NTU1_calc, Ntp=2, optimal=False)
        assert_allclose(P1, P1_calc, rtol=1E-6)
        
    with pytest.raises(Exception):
        NTU_from_P_H(P1=0.573, R1=1/3., Ntp=101) 



@pytest.mark.mpmath
def test_NTU_from_P_G():
    # 1 tube pass, random point
    R1 = 1.1
    NTU1 = 2
    P1_calc_orig = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1, Ntp=1)
    P1_expect = 0.5868787117241955
    assert_allclose(P1_calc_orig, P1_expect)
    NTU1_backwards = NTU_from_P_G(P1=P1_expect, R1=R1, Ntp=1)
    assert_allclose(NTU1, NTU1_backwards)
    

    # 2 tube pass, randompoint
    R1 = 1.1
    NTU1 = 2
    P1_calc_orig = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1, Ntp=2)
    P1_calc_orig
    P1_expect = 0.6110347802764724
    assert_allclose(P1_calc_orig, P1_expect)
    NTU1_backwards = NTU_from_P_G(P1=P1_expect, R1=R1, Ntp=2)
    assert_allclose(NTU1, NTU1_backwards)    
    

    # 2 tube pass, not optimal
    R1 = .1
    NTU1 = 2
    P1_calc_orig = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1, Ntp=2, optimal=False)
    P1_calc_orig
    P1_expect = 0.8121969945075509
    assert_allclose(P1_calc_orig, P1_expect)
    NTU1_backwards = NTU_from_P_G(P1=P1_expect, R1=R1, Ntp=2, optimal=False)
    assert_allclose(NTU1, NTU1_backwards)


    # Run the gamut testing all the solvers
    R1s = np.logspace(np.log10(2E-5), np.log10(1E2), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(1E2), 10000)
    seed(0)
    tot = 0
    for Ntp, optimal in zip([1, 2, 2], [True, True, False]):
        for i in range(100):
            R1 = float(choice(R1s))
            NTU1 = float(choice(NTU1s))
            try:
                P1 = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1, Ntp=Ntp, optimal=optimal)
                NTU1_calc = NTU_from_P_G(P1, R1, Ntp=Ntp, optimal=optimal)        
                P1_calc = temperature_effectiveness_TEMA_G(R1=R1, NTU1=NTU1_calc, Ntp=Ntp, optimal=optimal)
            except (ValueError, OverflowError, ZeroDivisionError, RuntimeError) as e:
                continue
            assert_allclose(P1, P1_calc)
            tot +=1
    assert tot > 270
    
    with pytest.raises(Exception):
        NTU_from_P_G(P1=.573, R1=1/3., Ntp=10)
    

@pytest.mark.mpmath
def test_NTU_from_P_J():
    # Run the gamut testing all the solvers
    R1s = np.logspace(np.log10(2E-5), np.log10(1E2), 10000)
    NTU1s = np.logspace(np.log10(1E-4), np.log10(1E2), 10000)
    seed(0)
    tot = 0
    for Ntp in [1, 2, 4]:
        for i in range(100):
            R1 = float(choice(R1s))
            NTU1 = float(choice(NTU1s))
            try:
                P1 = temperature_effectiveness_TEMA_J(R1=R1, NTU1=NTU1, Ntp=Ntp)
                NTU1_calc = NTU_from_P_J(P1, R1, Ntp=Ntp)        
                P1_calc = temperature_effectiveness_TEMA_J(R1=R1, NTU1=NTU1_calc, Ntp=Ntp)
            except (ValueError, OverflowError, ZeroDivisionError, RuntimeError) as e:
                continue
            assert_allclose(P1, P1_calc)
            tot +=1
    assert tot > 270
    # Actual individual understandable working test cases

    # 1 tube pass
    R1 = 1.1
    NTU1 = 3
    P1_calc_orig = temperature_effectiveness_TEMA_J(R1=R1, NTU1=NTU1, Ntp=1)
    P1_expect = 0.5996529947927913
    assert_allclose(P1_calc_orig, P1_expect)
    NTU1_backwards = NTU_from_P_J(P1=P1_expect, R1=R1, Ntp=1)
    assert_allclose(NTU1, NTU1_backwards)


    # 2 tube passes
    R1 = 1.1
    NTU1 = 2.7363888898379249
    P1_calc_orig = temperature_effectiveness_TEMA_J(R1=R1, NTU1=NTU1, Ntp=2)
    P1_expect = 0.53635261090479802
    assert_allclose(P1_calc_orig, P1_expect)
    # The exact P1 is slightly higher than that calculated as the upper limit 
    # of the pade approximation, so we multiply it by a small fraction
    NTU1_backwards = NTU_from_P_J(P1=P1_expect*(1-2E-9), R1=R1, Ntp=2)
    assert_allclose(NTU1, NTU1_backwards, rtol=1E-3)
    # Unfortunately the derivative is so large we can't compare it exactly


    # 4 tube passes
    R1 = 1.1
    NTU1 = 2.8702676768833268
    P1_calc_orig = temperature_effectiveness_TEMA_J(R1=R1, NTU1=NTU1, Ntp=4)
    P1_expect = 0.53812561986477236
    assert_allclose(P1_calc_orig, P1_expect)
    # The exact P1 is slightly higher than that calculated as the upper limit 
    # of the pade approximation, so we multiply it by a small fraction
    NTU1_backwards = NTU_from_P_J(P1=P1_expect*(1-1E-15), R1=R1, Ntp=4)
    assert_allclose(NTU1, NTU1_backwards)
    # The derivative is very large but the pade approximation is really good, ant it works


    with pytest.raises(Exception):
        # unsupported number of tube passes case
        NTU_from_P_J(P1=.57, R1=1/3., Ntp=10)


def test_NTU_from_P_plate():
    # 1 pass-1 pass counterflow
    NTU1 = 3.5
    R1 = 0.25
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=1, Np2=1)
    assert_allclose(P1_calc, 0.944668125335067)

    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=1, Np2=1)
    assert_allclose(NTU1, NTU1_calc)
    
    with pytest.raises(Exception):
        NTU_from_P_plate(P1=.10001, R1=10, Np1=1, Np2=1, counterflow=True)

    # 1 pass-1 pass parallelflow
    NTU1 = 3.5
    R1 = 0.25
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=1, Np2=1, counterflow=False)
    assert_allclose(P1_calc, 0.7899294862060529)
    
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=1, Np2=1, counterflow=False)
    assert_allclose(NTU1, NTU1_calc)
    
    with pytest.raises(Exception):
        NTU_from_P_plate(P1=.091, R1=10, Np1=1, Np2=1, counterflow=False)

    # 1-2 True True
    R1s = np.logspace(np.log10(2E-5), np.log10(10), 10000) # too high R1 causes overflows
    NTU1s = np.logspace(np.log10(1E-4), np.log10(99), 10000)
    
    tot = 0
    seed(0)
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=1, Np2=2)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=1, Np2=2)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=1, Np2=2)
        except (OverflowError, ValueError):
            continue
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot > 97
    
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=1, Np2=3)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=1, Np2=3)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=1, Np2=3)
        except (OverflowError, ValueError):
            continue
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot >= 99

    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=1, Np2=3, counterflow=False)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=1, Np2=3, counterflow=False)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=1, Np2=3, counterflow=False)
        except (OverflowError, ValueError):
            continue
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot >= 99

    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=1, Np2=4)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=1, Np2=4)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=1, Np2=4)
        except (OverflowError, ValueError):
            continue
        
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot >= 99
    
    # 2-2 pass cases

    # counterflow and not passes_counterflow
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=2, counterflow=True, passes_counterflow=False)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=2, counterflow=True, passes_counterflow=False)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=2, counterflow=True, passes_counterflow=False)
        except (OverflowError, ValueError):
            continue
        
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot >= 99
    
    # not counterflow and not passes_counterflow
    # random example
    NTU1 = 1.1
    R1 = .6
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=2, counterflow=False, passes_counterflow=False)
    assert_allclose(P1_calc, 0.5174719601105934)
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=2, Np2=2, counterflow=False, passes_counterflow=False)
    assert_allclose(NTU1, NTU1_calc)   
    # methodical test
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=2, counterflow=False, passes_counterflow=False)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=2, counterflow=False, passes_counterflow=False)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=2, counterflow=False, passes_counterflow=False)
        except ZeroDivisionError:
            continue
        assert_allclose(P1, P1_calc)
        tot +=1
    assert tot > 85

    # not counterflow and passes_counterflow
    # random example
    NTU1 = 1.1
    R1 = .6
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=2, counterflow=False, passes_counterflow=True)
    assert_allclose(P1_calc, 0.529647502598342)
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=2, Np2=2, counterflow=False, passes_counterflow=True)
    assert_allclose(NTU1, NTU1_calc)
    # methodical
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=2, counterflow=False, passes_counterflow=True)
        NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=2, counterflow=False, passes_counterflow=True)
        P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=2, counterflow=False, passes_counterflow=True)
        assert_allclose(P1, P1_calc)


    # 2-2 counterflow and passes_counterflow
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=2, counterflow=True, passes_counterflow=True)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=2, counterflow=True, passes_counterflow=True)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=2, counterflow=True, passes_counterflow=True)
        except (ValueError, ZeroDivisionError):
            continue
        tot +=1
        assert_allclose(P1, P1_calc)
    assert tot > 90
    
    
    # 2-3 counterflow - random example
    NTU1 = 1.1
    R1 = .6
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=3, counterflow=True)
    assert_allclose(P1_calc, 0.5696402802155714)
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=2, Np2=3, counterflow=True)
    assert_allclose(NTU1, NTU1_calc)
    # 2-3 counterflow - methodical
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=3, counterflow=True)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=3, counterflow=True)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=3, counterflow=True)
        except (ValueError, ZeroDivisionError):
            continue
        tot +=1
        assert_allclose(P1, P1_calc, rtol=5E-4)
    assert tot > 85
    
    # 2-3 parallelflow - random example
    NTU1 = 1.1
    R1 = .6
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=3, counterflow=False)
    assert_allclose(P1_calc, 0.5272339114328507)
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=2, Np2=3, counterflow=False)
    assert_allclose(NTU1, NTU1_calc)
    # 2-3 parallelflow - methodical (all work for given range)
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=3, counterflow=False)
        NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=3, counterflow=False)
        P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=3, counterflow=False)
        assert_allclose(P1, P1_calc)

    
    # 2-4 counterflow - random example
    NTU1 = 1.1
    R1 = .6
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=4, counterflow=True)
    assert_allclose(P1_calc, 0.5717083161054717)
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=2, Np2=4, counterflow=True)
    assert_allclose(NTU1, NTU1_calc)
    # 2-4 counterflow - methodical
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=4, counterflow=True)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=4, counterflow=True)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=4, counterflow=True)
        except (ValueError, ZeroDivisionError):
            continue
        tot +=1
        assert_allclose(P1, P1_calc)
    assert tot > 95
    
    # 2-4 parallelflow - random example
    NTU1 = 1.1
    R1 = .6
    P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=4, counterflow=False)
    assert_allclose(P1_calc, 0.5238412695944656)
    NTU1_calc = NTU_from_P_plate(P1=P1_calc, R1=R1, Np1=2, Np2=4, counterflow=False)
    assert_allclose(NTU1, NTU1_calc)
    # 2-4 counterflow - methodical
    tot = 0
    for i in range(100):
        R1 = float(choice(R1s))
        NTU1 = float(choice(NTU1s))
        try:
            P1 = temperature_effectiveness_plate(R1=R1, NTU1=NTU1, Np1=2, Np2=4, counterflow=False)
            NTU1_calc = NTU_from_P_plate(P1, R1, Np1=2, Np2=4, counterflow=False)
            P1_calc = temperature_effectiveness_plate(R1=R1, NTU1=NTU1_calc, Np1=2, Np2=4, counterflow=False)
        except (ValueError, ZeroDivisionError):
            continue
        tot +=1
        assert_allclose(P1, P1_calc)
    assert tot > 95


    # Backwards, only one example in the tests
    # No real point in being exhaustive
    NTU1 = NTU_from_P_plate(P1=0.5743514352720835, R1=1/3., Np1=3, Np2=1)
    assert_allclose(NTU1, 1)
    
    # Bad number of plates
    with pytest.raises(Exception):
        NTU_from_P_plate(P1=0.5743, R1=1/3., Np1=3, Np2=13415151213) 

def test_DBundle_min():
    assert_allclose(DBundle_min(0.0254), 1)
    assert_allclose(DBundle_min(0.005), .1)
    assert_allclose(DBundle_min(0.014), .3)
    assert_allclose(DBundle_min(0.015), .5)
    assert_allclose(DBundle_min(.1), 1.5)

def test_shell_clearance():
    assert_allclose(shell_clearance(DBundle=1.245), 0.0064)
    assert_allclose(shell_clearance(DBundle=4), 0.011)
    assert_allclose(shell_clearance(DBundle=.2), .0032)
    assert_allclose(shell_clearance(DBundle=1.778), 0.0095)
    
    assert_allclose(shell_clearance(DShell=1.245), 0.0064)
    assert_allclose(shell_clearance(DShell=4), 0.011)
    assert_allclose(shell_clearance(DShell=.2), .0032)
    assert_allclose(shell_clearance(DShell=1.778), 0.0095)
    
    with pytest.raises(Exception):
        shell_clearance()

def test_L_unsupported_max():
    assert_allclose(L_unsupported_max(Do=.0254, material='CS'), 1.88)
    assert_allclose(L_unsupported_max(Do=.0253, material='CS'), 1.753)
    assert_allclose(L_unsupported_max(Do=1E-5, material='CS'), 0.66)
    assert_allclose(L_unsupported_max(Do=.00635, material='CS'), 0.66)
    
    assert_allclose(L_unsupported_max(Do=.00635, material='aluminium'), 0.559)
    
    with pytest.raises(Exception):
        L_unsupported_max(Do=.0254, material='BADMATERIAL')
        
    # Terribly pessimistic
    assert_allclose(L_unsupported_max(Do=10, material='CS'), 3.175)