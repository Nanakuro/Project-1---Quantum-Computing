#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 19:14:23 2019

@author: Minh Nguyen
"""

import numpy as np
import quantum_sim_module as qs
from matplotlib import pyplot as plt
from copy import deepcopy

def PlotPhi(num_t_wires, my_phi, state):
    myState = state[:]
    top_wires = num_t_wires
    phi = my_phi
    count = 10000
    measurements = []
    final_state = qs.PhaseEstimation(top_wires, 'CRz', myState, phase=phi)
    for c in range(count):
        measure = qs.Measure(final_state)
        theta = sum([int(s)/2**(i+1) for i,s in enumerate(measure[:top_wires])])
        measurements.append(theta)
        
    save_state_name = 'Psi0+Psi1'
    fig = plt.figure()
    plt.title(r'Phase Estimation Histogram $|%s\rangle$ ,$\phi = %.1f$' % (save_state_name, my_phi))
    plt.xlabel(r'Theta')
    plt.ylabel(r'Counts')
    plt.hist(measurements, bins=(2**num_t_wires))
    plt.show()
    #plt.savefig('measure_phase_est_CRz_%s.png' % save_state_name, dpi=300)

def PlotPhaseEst(top_wires, phi_list, init_state):
    amplitudes = []
    norm_phi_list = phi_list
    num_top_wires = top_wires

    for norm_phi in norm_phi_list:
        myState = deepcopy(init_state)
        phi = norm_phi * 2*np.pi
        est_state = qs.PhaseEstimation(num_top_wires, 'CPhase', myState, phase=phi)
        amplitudes.append(qs.GetMaxTheta(num_top_wires, est_state))
        #amplitudes.append(PhaseEstimation01(num_top_wires, phi, myState))
        #amplitudes.append(PhaseEstimation001(num_top_wires, phi, myState))
    
    save_state_name = 'Psi0+Psi1'
    fig = plt.figure()
    plt.xlabel(r'Normalized phases ($\phi / 2\pi$)')
    plt.ylabel(r'Theta ($\theta_j$)')
    plt.plot(norm_phi_list, amplitudes)
    plt.title(r'Phase Estimation $|%s\rangle$' % save_state_name)
    plt.show()
    #plt.savefig('phase_estimation_%s.png' % save_state_name, dpi=300)

'''
t_wires = 6
#myState = GetStateWithBottomWireCircuit(t_wires, 'my_random.circuit')
#myState = [[0.3**0.5, '0000000'],[0.7**0.5, '0000001']]
#myState = [[1.0,'0000001']]
my_state = [[0.3**(0.5),'0000000'],[0.7**(0.5),'0000001']]
phis = np.linspace(0,1,100,endpoint=False)
phase = 0.5
#PlotPhaseEst(t_wires, phis, my_state)
PlotPhi(t_wires, phase, my_state)
'''