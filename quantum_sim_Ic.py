#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:36:41 2019

@author: Minh Nguyen
"""

import numpy as np
import quantum_sim_module as qs
from matplotlib import pyplot as plt

def runCircuitIc(stateFile):
    n_wires, in_circuit = qs.ReadInput(stateFile)
    list_of_gates = []
    for inp in in_circuit:
        if inp[0] == 'H':
            list_of_gates.append(qs.HadamardSparse(int(inp[1]),n_wires))
        elif inp[0] == 'P':
            list_of_gates.append(qs.PhaseSparse(int(inp[1]),n_wires,float(inp[2])))
        elif inp[0] == 'CNOT':
            list_of_gates.append(qs.CNOTSparse(int(inp[1]),int(inp[2]), n_wires))
    in_vector = qs.StateToVec(qs.GetInputState(n_wires, in_circuit))
    out_vector = qs.computeState(in_vector,list_of_gates)
    tol = 1E-9
    out_vector.real[np.abs(out_vector.real) < tol] = 0.0
    out_vector.imag[np.abs(out_vector.imag) < tol] = 0.0
    out_state = qs.VecToState(out_vector)
    if in_circuit[-1][0] == 'MEASURE':
        count = 10000
        measurements = []
        for c in range(count):
            measure = qs.Measure(out_state)
            index = int(measure,2)
            measurements.append(index)
        fig = plt.figure()
        plt.hist(measurements,bins=2**n_wires)
        plt.title('Measurement histogram')
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.savefig('measure_circuit_Ic.png')
    else:
        #qs.PrettyPrintBinary(out_state)
        qs.PrettyPrintInteger(out_state)
