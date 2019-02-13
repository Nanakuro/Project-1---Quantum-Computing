#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 19:14:23 2019

@author: Minh Nguyen
"""

import numpy as np
import quantum_sim_module as qs
import quantum_sim_II as qII
from matplotlib import pyplot as plt

def PhaseEstimation(top_wires, phase, state):
    bot_wires = len(state[0][1]) - top_wires

    for i in range(top_wires):
        qs.Hadamard(i, state)
    
    qs.PhaseSeries(top_wires, phase, state)
    qs.rmDupe(state)
    
    qs.ShiftRight(bot_wires, state)
    qs.invQFT(top_wires, state)
    qs.ShiftLeft(bot_wires, state)
    qs.reverseState(top_wires, state)
    
    qs.rmDupe(state)
    return state

def PlotPhi(num_t_wires, my_phi, state):
    myState = state[:]
    top_wires = num_t_wires
    phi = my_phi
    count = 10000
    measurements = []
    final_state = PhaseEstimation(top_wires, phi, myState)
    print(final_state)
    for c in range(count):
        measure = qs.Measure(final_state)
        theta = sum([int(s)/2**(i+1) for i,s in enumerate(measure[:top_wires])])
        measurements.append(theta)
        
    save_state_name = state[0][1] # 'Psi0+Psi1'
    fig = plt.figure()
    plt.title(r'Phase Estimation Histogram $|%s\rangle$ ,$\phi = %.1f$' % (save_state_name, my_phi))
    plt.xlabel(r'Theta')
    plt.ylabel(r'Counts')
    plt.hist(measurements, bins=(2**num_t_wires))
    plt.show()
    #plt.savefig('measure_phase_est_%s.png' % save_state_name, dpi=300)

def PlotPhaseEst(top_wires, phi_list, state):
    amplitudes = []
    norm_phi_list = phi_list
    initState = state[:]
    num_top_wires = top_wires

    for norm_phi in norm_phi_list:
        myState = initState[:]
        phi = norm_phi * 2*np.pi
        amplitudes.append(GetMaxTheta(num_top_wires, PhaseEstimation(num_top_wires, phi, myState)))
        #amplitudes.append(PhaseEstimation01(num_top_wires, phi, myState))
        #amplitudes.append(PhaseEstimation001(num_top_wires, phi, myState))
    
    save_state_name = '01'
    fig = plt.figure()
    plt.xlabel(r'Normalized phases ($\phi / 2\pi$)')
    plt.ylabel(r'Theta ($\theta_j$)')
    plt.plot(norm_phi_list, amplitudes)
    plt.title(r'Phase Estimation $|%s\rangle$' % save_state_name)
    plt.show()
    #plt.savefig('phase_estimation_%s.png' % save_state_name, dpi=300)

def GetMaxTheta(top_wires, state):
    amp_list = [ abs(s[0]) for s in state ]
    max_state = state[amp_list.index(max(amp_list))][1]
    return sum([int(s)/2**(i+1) for i,s in enumerate(max_state[:top_wires])])

def GetStateWithBottomWireCircuit(top_wires, fileName):
    n_wires, in_circuit = qs.ReadInput(fileName)
    the_state = qs.GetInputState(n_wires,in_circuit)
    for inp in in_circuit:
        if inp[0] == 'P':           qs.Phase(int(inp[1]), float(inp[2]), the_state)
        elif inp[0] == 'NOT':       qs.NOT(int(inp[1]), the_state)
    qs.rmDupe(the_state)
    State = [ [s[0], s[1].zfill(top_wires+len(s[1]))] for s in the_state ]
    return State
    
t_wires = 6
#myState = GetStateWithBottomWireCircuit(t_wires, 'my_random.circuit')
#myState = [[0.3**0.5, '0000000'],[0.7**0.5, '0000001']]
#myState = [[1.0,'0000001']]
myState = [[1.0,'0000001']]
phase = 0.5
PlotPhi(t_wires, phase, myState)