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
    circuit_matrix = qs.CircuitMatrix(n_wires, in_circuit)
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
        plt.title('Measurement histogram - Ia (%d wires)' % n_wires)
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.savefig('measure_circuit_Ia_%d_wires.png' % n_wires)
    else:
        #qs.PrettyPrintBinary(out_state)
        qs.PrettyPrintInteger(out_state)
runCircuitIa('input.circuit')
'''
list_of_circuit_files = ['rand5.circuit',
                         'rand6.circuit',
                         'rand7.circuit',
                         'rand8.circuit',
                         'rand9.circuit',
                         'rand10.circuit',
                         'rand11.circuit',
                         'rand12.circuit']
print('Running Ia...')
for i in range(len(list_of_circuit_files)):
    file = list_of_circuit_files[i]
    runCircuitIa(file)
    if i == len(list_of_circuit_files)-1:
        print('Done')
        break
    user_flag = input('Continue to run %s? ' % list_of_circuit_files[i+1][:-8])
    if user_flag not in ['y','yes','1']:
        break
'''