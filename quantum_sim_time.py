#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 00:54:33 2019

@author: Minh Nguyen
"""

import time
import quantum_sim_Ia as Ia
import quantum_sim_Ib as Ib
import quantum_sim_Ic as Ic

start_Ia = time.time()
Ia.runCircuitIa('rand.circuit')
Ia.runCircuitIa('measure.circuit')
Ia.runCircuitIa('input.circuit')
timeIa = time.time() - start_Ia

start_Ib = time.time()
Ib.runCircuitIb('rand.circuit')
Ib.runCircuitIb('measure.circuit')
Ib.runCircuitIb('input.circuit')
timeIb = time.time() - start_Ib

start_Ic = time.time()
Ic.runCircuitIc('rand.circuit')
Ic.runCircuitIc('measure.circuit')
Ic.runCircuitIc('input.circuit')
timeIc = time.time() - start_Ic

print('Quantum Simulator Ia: %g s' % timeIa)
print('Quantum Simulator Ib: %g s' % timeIb)
print('Quantum Simulator Ic: %g s' % timeIc)
