#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:26:49 2019

@author: Minh Nguyen
"""

import numpy as np
import quantum_sim_module as qs
from matplotlib import pyplot as plt

def runCircuitIa(stateFile):
    n_wires, in_circuit = qs.ReadInput(stateFile)
    circuit_matrix = qs.UnitaryMatrix(n_wires, in_circuit)
    in_vector = qs.StateToVec(qs.GetInputState(n_wires, in_circuit))
    out_vector = circuit_matrix @ in_vector
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
        plt.savefig('measure_circuit_Ia.png')
    else:
        #qs.PrettyPrintBinary(out_state)
        qs.PrettyPrintInteger(out_state)
        
