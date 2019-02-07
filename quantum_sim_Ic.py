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
    in_vector = qs.StateToVec(qs.GetInputState(n_wires, in_circuit))
    for inp in in_circuit:
        if inp[0] == 'H':
            in_vector = qs.HadamardSparse(int(inp[1]),n_wires).dot(in_vector)
        elif inp[0] == 'P':
            in_vector = qs.PhaseSparse(int(inp[1]),n_wires,float(inp[2])).dot(in_vector)
        elif inp[0] == 'CNOT':
            in_vector = qs.CNOTSparse(int(inp[1]),int(inp[2]), n_wires).dot(in_vector)
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
        plt.title('Measurement histogram - Ic (%d wires)' % n_wires)
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.savefig('measure_circuit_Ic_%d_wires.png' % n_wires)
    else:
        #qs.PrettyPrintBinary(out_state)
        qs.PrettyPrintInteger(out_state)
'''
list_of_circuit_files = ['rand5.circuit',
                         'rand6.circuit',
                         'rand7.circuit',
                         'rand8.circuit',
                         'rand9.circuit',
                         'rand10.circuit',
                         'rand11.circuit',
                         'rand12.circuit',
                         'rand20.circuit']
print('Running Ic...')
for i in range(len(list_of_circuit_files)):
    file = list_of_circuit_files[i]
    runCircuitIc(file)
    if i == len(list_of_circuit_files)-1:
        print('Done')
        break
    user_flag = input('Continue to run %s?' % list_of_circuit_files[i+1][:-8])
    if user_flag not in ['y','yes','1']:
        break
'''