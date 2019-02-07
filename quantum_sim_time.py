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
from matplotlib import pyplot as plt

def timeSimulators(fileList):
    timeIa_list = []
    timeIb_list = []
    timeIc_list = []
    
    number_of_wires_list = []
    for f in fileList:
        circuit_wires = int(f[4:-8])
        number_of_wires_list.append(circuit_wires)

        start_Ic = time.time()
        Ic.runCircuitIc(f)
        timeIc = time.time() - start_Ic
        timeIc_list.append(timeIc)

        if circuit_wires == 20:
            break
        
        start_Ia = time.time()
        Ia.runCircuitIa(f)
        timeIa = time.time() - start_Ia
        timeIa_list.append(timeIa)

        start_Ib = time.time()
        Ib.runCircuitIb(f)
        timeIb = time.time() - start_Ib
        timeIb_list.append(timeIb)
    
    fig = plt.figure()
    
    if len(number_of_wires_list) == len(timeIa_list):
        plt.plot(number_of_wires_list, timeIa_list, '-o',label='Ia')
        plt.plot(number_of_wires_list, timeIb_list, '-o',label='Ib')
    elif len(number_of_wires_list) - len(timeIa_list) == 1:
        plt.plot(number_of_wires_list[:-1], timeIa_list, '-o',label='Ia')
        plt.plot(number_of_wires_list[:-1], timeIb_list, '-o',label='Ib')
    plt.plot(number_of_wires_list, timeIc_list, '-o',label='Ic')
    plt.title('Ia vs Ib vs Ic TIME')
    plt.xlabel('Number of wires')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='upper right')
    plt.savefig('time_comparison.png')
        


list_of_circuit_files = ['rand5.circuit',
                         'rand6.circuit',
                         'rand7.circuit',
                         'rand8.circuit',
                         'rand9.circuit',
                         'rand10.circuit',
                         'rand11.circuit',
                         'rand12.circuit',
                         'rand20.circuit']

timeSimulators(list_of_circuit_files)

