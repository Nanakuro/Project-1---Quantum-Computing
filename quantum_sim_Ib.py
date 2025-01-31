#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:17:03 2019

@author: Minh Nguyen
"""

import numpy as np
import quantum_sim_module as qs
from matplotlib import pyplot as plt

def runCircuitIb(stateFile):
    n_wires, in_circuit = qs.ReadInput(stateFile)
    in_vector = qs.StateToVec(qs.GetInputState(n_wires, in_circuit))
    for inp in in_circuit:
        if inp[0] == 'H':
            in_vector = qs.HadamardArray(int(inp[1]), n_wires) @ in_vector
        elif inp[0] == 'P':
            in_vector = qs.PhaseArray(int(inp[1]), n_wires, float(inp[2])) @ in_vector
        elif inp[0] == 'CNOT':
            in_vector = qs.CNOTArray(int(inp[1]),int(inp[2]), n_wires) @ in_vector
    tol = 1E-9
    in_vector.real[np.abs(in_vector.real) < tol] = 0.0
    in_vector.imag[np.abs(in_vector.imag) < tol] = 0.0
    out_state = qs.VecToState(in_vector)
    if in_circuit[-1][0] == 'MEASURE':
        count = 10000
        measurements = []
        for c in range(count):
            measure = qs.Measure(out_state)
            index = int(measure,2)
            measurements.append(index)
        fig = plt.figure()
        plt.hist(measurements,bins=2**n_wires)
        plt.title('Measurement histogram - Ib (%d wires)' % n_wires)
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.savefig('measure_circuit_Ib_%d_wires.png' % n_wires)
    else:
        #qs.PrettyPrintBinary(out_state)
        qs.PrettyPrintInteger(out_state)
runCircuitIb('rand.circuit')

'''
list_of_circuit_files = ['rand5.circuit',
                         'rand6.circuit',
                         'rand7.circuit',
                         'rand8.circuit',
                         'rand9.circuit',
                         'rand10.circuit',
                         'rand11.circuit',
                         'rand12.circuit']
print('Running Ib...')
for i in range(len(list_of_circuit_files)):
    file = list_of_circuit_files[i]
    runCircuitIb(file)
    if i == len(list_of_circuit_files)-1:
        print('Done')
        break
    user_flag = input('Continue to run %s? ' % list_of_circuit_files[i+1][:-8])
    if user_flag not in ['y','yes','1']:
        break
'''