#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 20:36:24 2019

@author: Minh Nguyen
"""

import numpy as np
import quantum_sim_module as qs
from matplotlib import pyplot as plt


def runCircuitIIwithoutCompile(circuitFile):
    n_wires, in_circuit = qs.ReadInput(circuitFile)
    the_state = qs.GetInputState(n_wires,in_circuit)
    for inp in in_circuit:
        if inp[0] == 'H':           qs.Hadamard(int(inp[1]), the_state)
        elif inp[0] == 'P':         qs.Phase(int(inp[1]), float(inp[2]), the_state)
        elif inp[0] == 'CNOT':      qs.CNOT(int(inp[1]), int(inp[2]), the_state)
        elif inp[0] == 'NOT':       qs.NOT(int(inp[1]), the_state)
        elif inp[0] == 'Rz':        qs.Rz(int(inp[1]), float(inp[2]), the_state)
        elif inp[0] == 'CRz':       qs.CRz(int(inp[1]), int(inp[2]), float(inp[3]), the_state)
        elif inp[0] == 'CPhase':    qs.CPhase(int(inp[1]), int(inp[2]), float(inp[3]), the_state)
        elif inp[0] == 'SWAP':      qs.SWAP(int(inp[1]), int(inp[2]), the_state)
        
    tol = 1E-6
    for s in the_state:
        if abs(round(s[0].real)-s[0].real) < tol: s[0] = complex(round(s[0].real),s[0].imag)
        if abs(round(s[0].imag)-s[0].imag) < tol: s[0] = complex(s[0].real,round(s[0].imag))
    
    if in_circuit[-1][0] == 'MEASURE':
        count = 10000
        measurements = []
        for c in range(count):
            measure = qs.Measure(the_state)
            index = int(measure,2)
            measurements.append(index)
        fig = plt.figure()
        plt.hist(measurements,bins=2**n_wires)
        plt.title('Measurement histogram - Ic (%d wires)' % n_wires)
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.savefig('measure_circuit_Ic_%d_wires.png' % n_wires)
    else:
        #qs.PrettyPrintBinary(the_state)
        qs.PrettyPrintInteger(the_state)


def runCircuitIIwithCompile(inFile):
    runCircuitIIwithoutCompile('basic_' + inFile)
    
runCircuitIIwithoutCompile('non_atomic_gates.circuit')