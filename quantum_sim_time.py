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

def runSimulators(fileList):
    timeIa_list = []
    timeIb_list = []
    timeIc_list = []
    
    
    start_Ia = time.time()
    for f in fileList:  Ia.runCircuitIa(f)
    #Ia.runCircuitIa('rand.circuit')
    #Ia.runCircuitIa('measure.circuit')
    #Ia.runCircuitIa('input.circuit')
    timeIa = time.time() - start_Ia
    timeIa_list.append(timeIa)

    start_Ib = time.time()
    for f in fileList:  Ib.runCircuitIb(f)
    #Ib.runCircuitIb('rand.circuit')
    #Ib.runCircuitIb('measure.circuit')
    #Ib.runCircuitIb('input.circuit')
    timeIb = time.time() - start_Ib
    timeIb_list.append(timeIb)
    
    start_Ic = time.time()
    for f in fileList:  Ic.runCircuitIc(f)
    #Ic.runCircuitIc('rand.circuit')
    #Ic.runCircuitIc('measure.circuit')
    #Ic.runCircuitIc('input.circuit')
    timeIc = time.time() - start_Ic
    timeIc_list.append(timeIc)
    
    
    
    print(type(timeIa))
    print('Quantum Simulator Ia: %g s' % timeIa)
    print('Quantum Simulator Ib: %g s' % timeIb)
    print('Quantum Simulator Ic: %g s' % timeIc)
        


list_of_circuit_files = ['rand5.circuit',
                         'rand6.circuit',
                         'rand7.circuit',
                         'rand8.circuit',
                         'rand9.circuit',
                         'rand10.circuit',
                         'rand11.circuit',
                         'rand12.circuit',
                         'rand20.circuit']#,'input.circuit']#,'measure.circuit']
runSimulators(list_of_circuit_files)

