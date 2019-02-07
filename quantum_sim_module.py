#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:17:03 2019

@author: Minh Nguyen
"""

import numpy as np
import scipy.sparse as sp
from random import choices

def PrettyPrintBinary(state):
    i = 1
    for s in state:
        if i == len(state):
            print('({:g}) |{:s}>'.format(s[0], s[1]))
        else:
            print('({:g}) |{:s}>'.format(s[0], s[1]),end=' + ')
        i += 1

def PrettyPrintInteger(state):
    i = 1
    for s in state:
        num = int(s[1],2)   # Convert binary string to base-10 integer
        if i == len(state):
            print('({:g}) |{:g}>'.format(s[0], num))
        else:
            print('({:g}) |{:g}>'.format(s[0], num),end=' + ')
        i += 1
    
def StateToVec(state):  
    vector = np.zeros(2**len(state[0][1]))
    vector = [ float(v) for v in vector]
    for s in state:
        i = int(s[1],2)
        vector[i] = s[0]
    return vector

def VecToState(vector):
    state = []
    l = 0
    for i in range(len(vector)):
        v = vector[i]
        if v != 0.0:
            basis = "{0:b}".format(i)   # Convert base-10 integer to binary string
            state.append([v,basis])
            l = len(basis)
    for s in state:
        s[1] = s[1].zfill(l)
    #state = [tuple(s) for s in state]
    return state
         
############################## Quantum Simulator Ia ###########################

def HadamardArray(wire,total):     # Apply Hadamard matrix to wire i out of k wires
    Had = 1/np.sqrt(2) * np.array([[1,1],
                                        [1,-1]])
    matrix = np.identity(2)
    if wire == total-1:     matrix = Had
    w = 1
    while w < total:
        if w == total-wire-1:
            matrix = np.kron(Had, matrix)
        else:
            matrix = np.kron(np.identity(2), matrix)
        w += 1
    
    return matrix

def PhaseArray(wire,total,phi):
    P = np.array([[1,           0],
                      [0,   np.exp(phi*1.j)]])
    matrix = np.identity(2)
    if wire == total-1:     matrix = P
    w = 1
    while w < total:
        if w == total-wire-1:
            matrix = np.kron(P, matrix)
        else:
            matrix = np.kron(np.identity(2), matrix)
        w += 1
    return matrix

# Currently only apply to target wires that is next to a control wire.
def CNOTArray(control,target,total):
    if total < 2:
        print('Not enough number of qubits to implement CNOT gate.')
    elif control == target:
        print('Invalid placement of CNOT gate (control wire and target wire must be different).')
    else:
        
        CNOTdown = np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,0,1],
                             [0,0,1,0]])    
    
        CNOTup = np.array([[1,0,0,0],
                           [0,0,0,1],
                           [0,0,1,0],
                           [0,1,0,0]])
    
        w = 0
        if control == total-1:
            matrix = CNOTup
            w += 1
        elif target == total-1:
            matrix = CNOTdown
            w += 1
        else:
            matrix = np.identity(2)
        w += 1
        
        while w < total:
            if target < control and w == total-1-control:
                matrix = np.kron(CNOTup, matrix)
                w += 1
            elif target > control and w == total-1-target:
                matrix = np.kron(CNOTdown, matrix)
                w += 1
            else:
                matrix = np.kron(np.identity(2), matrix)
            w += 1
        return matrix

def UnitaryMatrix(num_wires,input_circuit):
    uni_matrix = np.identity(2**num_wires)
    for inp in input_circuit:
        if inp[0] == 'H':
            uni_matrix = HadamardArray(int(inp[1]),num_wires) @ uni_matrix
        elif inp[0] == 'P':
            uni_matrix = PhaseArray(int(inp[1]),num_wires,float(inp[2])) @ uni_matrix
        elif inp[0] == 'CNOT':
            uni_matrix = CNOTArray(int(inp[1]),int(inp[2]), num_wires) @ uni_matrix
    tol = 1E-9
    uni_matrix.real[np.abs(uni_matrix.real) < tol] = 0
    uni_matrix.imag[np.abs(uni_matrix.imag) < tol] = 0
    return uni_matrix
     
def ReadInput(fileName):
    myInput_lines = open(fileName).readlines()
    myInput = []
    numberOfWires = int(myInput_lines[0].strip())
    for line in myInput_lines[1:]:
        myInput.append(line.split())
    return (numberOfWires,myInput)

def Measure(state):
    result = []
    probability = []
    for s in state:
        probability.append(np.abs(s[0])**2)
        result.append(s[1])
    total_probability = 0
    for p in probability: total_probability += p
    if np.abs(total_probability - 1) > 10**6:
        print('Total probability does not add up to 1.')
        return
    return choices(result,weights=probability)[0]

def GetInputState(numberOfWires,input_circuit):
    if input_circuit[0][0] == 'INITSTATE':
        if input_circuit[0][1] == 'FILE':
            vec = []
            with open('%s.dms' % input_circuit[0][2]) as f:
                for line in f:
                    l = line.strip().split()
                    v = '%s+%sj' % (l[0],l[1])
                    v = v.replace(' ','')   # Clear white spaces
                    v = v.replace('+-','-') # Change '+-' to just '-'
                    vec.append(complex(v))
            return VecToState(vec)
        elif input_circuit[0][1] == 'BASIS':
            state = [[1,input_circuit[0][2].strip('|').strip('>')]]
            return state
    else:
        return [[1,'0'*numberOfWires]]

############################ Quantum Simulator Ic #############################

def HadamardSparse(wire,total):     # Apply Hadamard matrix to wire i out of k wires
    Hadamard = sp.csr_matrix(1/np.sqrt(2) * np.array([[1,1],
                                                     [1,-1]]), dtype='complex')
    if wire == 0:
        matrix = Hadamard
    else:
        matrix = sp.csr_matrix(sp.identity(2,dtype='complex'))
    for w in range(1,total):
        if w == total-wire-1:
            matrix = sp.kron(Hadamard.tocsr(),matrix.tocsr(),'csr')
        else:
            matrix = sp.kron(sp.csr_matrix(sp.identity(2,dtype='complex')).tocsr(),matrix.tocsr(),'csr')
    return matrix.tocsr()

def PhaseSparse(wire,total,phi):
    Phase = sp.csr_matrix([[1,           0],
                           [0,   np.exp(phi*1.j)]])
    if wire == 0:
        matrix = Phase
    else:
        matrix = sp.csr_matrix(sp.identity(2,dtype='complex'))
    for w in range(1,total):
        if w == total-wire-1:
            matrix = sp.kron(Phase,matrix,'csr')
        else:
            matrix = sp.kron(sp.csr_matrix(sp.identity(2,dtype='complex')),matrix,'csr')
    return matrix.tocsr()

# Currently only apply to target wires that is next to a control wire.
def CNOTSparse(control,target,total):
    if total < 2:
        print('Not enough number of qubits to implement CNOT gate.')
    elif control == target:
        print('Invalid placement of CNOT gate (control wire and target wire must be different).')
    else:
        CNOTdown = sp.csr_matrix([[1,0,0,0],
                                  [0,1,0,0],
                                  [0,0,0,1],
                                  [0,0,1,0]])
        CNOTup = sp.csr_matrix([[1,0,0,0],
                                [0,0,0,1],
                                [0,0,1,0],
                                [0,1,0,0]])
        w = 0
        if control == 0:
            matrix = CNOTup
            w += 1
        elif target == 0:
            matrix = CNOTdown
            w += 1
        else:
            matrix = sp.csr_matrix(sp.identity(2, dtype='complex'))
        w += 1

        while w < total:
            if target < control and w == total-1-control:
                matrix = sp.kron(CNOTup.tocsr(), matrix.tocsr(),'csr')
                w += 1
            elif target > control and w == total-1-target:
                matrix = sp.kron(CNOTdown.tocsr(), matrix.tocsr(),'csr')
                w += 1
            else:
                matrix = sp.kron(sp.csr_matrix(sp.identity(2,dtype='complex')).tocsr(), matrix.tocsr(),'csr')
            w += 1
        return matrix.tocsr()

############################ Quantum Simulator II #############################

def rmDupe(state):
    for i,s_main in enumerate(state):
        for j,s_check in enumerate(state[i+1:]):
            if s_main[1] == s_check[1]:
                s_main[0] += s_check[0]
                del(state[i+j+1])
        if s_main[0] == 0.0:
            del(state[i])
    
        
def Hadamard(wire, state):
    ind = 0
    while ind < len(state):
        amp = state[ind][0]
        qubit = state[ind][1]
        
        if qubit[wire] == '0':
            amp /= 2**(0.5)
            # Change amplitude of state '...0...'
            state[ind][0] = amp
            
            # Add a state '...1...'
            ind += 1
            qubit = qubit[:wire] + '1' + qubit[wire+1:]
            state.insert(ind,[amp,qubit])
            #print(state)
        elif qubit[wire] == '1':
            amp /= 2**(0.5)
            # Change amplitude of state '...1...'
            state[ind][0] = -amp
            
            # Add a state '...0...'
            ind += 1
            qubit = qubit[:wire] + '0' + qubit[wire+1:]
            state.insert(ind,[amp,qubit])
            
        ind += 1
    rmDupe(state)
    
def Phase(wire, phi, state):
    for s in state:
        if s[1][wire] == '1':
            s[0] *= np.exp(phi*1.j)
    rmDupe(state)

def CNOT(control, target, state):
    for i,s in enumerate(state):
        qubit = s[1]
        if qubit[control] == '1':
            if qubit[target] == '0':
                state[i][1] = qubit[:target] + '1' + qubit[target+1:]
            elif qubit[target] == '1':
                state[i][1] = qubit[:target] + '0' + qubit[target+1:]
    rmDupe(state)
    
############################## Non-Atomic Gates ###############################

def NOT(wire, state):
    Hadamard(wire, state)
    Phase(wire, np.pi, state)
    Hadamard(wire, state)

def Rz(wire, phi, state):
    Phase(wire, phi/2.0, state)
    NOT(wire, state)
    Phase(wire, -phi/2.0, state)
    NOT(wire, state)

def CRz(control, target, phi, state):
    Phase(target, phi/2.0, state)
    CNOT(control, target, state)
    Phase(target, -phi/2.0, state)
    CNOT(control, target, state)

def CPhase(control, target, phi, state):
    CRz(control, target, phi, state)
    Phase(control, phi/2.0)

def SWAP(w1, w2, state):
    CNOT(w1, w2, state)
    CNOT(w2, w1, state)
    CNOT(w1, w2, state)

################################ Pre-Compiling ################################

def compileNOT(NOT_inputs):
    # NOT_inputs = ['NOT', wire]
    wNOT = NOT_inputs[1]
    basic_NOT = ['H %s' % wNOT,
                 'P %s %.9f' % (wNOT,np.pi),
                 'H %s' % wNOT]
    return '\n'.join(basic_NOT)

def compileRz(Rz_inputs):
    # Rz_inputs = ['Rz', wire, phase]
    wRz = Rz_inputs[1]
    p = Rz_inputs[2]
    basic_Rz = ['P %s %s' % (wRz,float(p)/2.0),
                compileNOT(['NOT',wRz]),
                'P %s %s' % (wRz,-float(p)/2.0),
                compileNOT(['NOT',wRz])]
    return '\n'.join(basic_Rz)

def compileCRz(CRz_inputs):
    # CRz_inputs = ['CRz', control_wire, target_wire, phase]
    c_wire = CRz_inputs[1]
    t_wire = CRz_inputs[2]
    p = CRz_inputs[3]
    
    basic_CRz = ['P %s %s' % (t_wire,float(p)/2.0),
                 'CNOT %s %s' % (c_wire, t_wire),
                 'P %s %s' % (t_wire,-float(p)/2.0),
                 'CNOT %s %s' % (c_wire, t_wire)]
    return '\n'.join(basic_CRz)

def compileCPhase(CPhase_inputs):
    # CPhase_inputs = ['CPHASE', control_wire, target_wire, phase]
    control = CPhase_inputs[1]
    target = CPhase_inputs[2]
    phase = CPhase_inputs[3]
    
    basic_CPhase = [compileCRz(['CPhase',control, target, phase]),
                    'P %s %s' % (control, float(phase)/2.0)]
    return '\n'.join(basic_CPhase)

def compileSWAP(SWAP_inputs):
    # SWAP_inputs = ['SWAP', wire1, wire2]
    w1 = SWAP_inputs[1]
    w2 = SWAP_inputs[2]
    
    basic_SWAP = ['CNOT %s %s' % (w1, w2),
                  'CNOT %s %s' % (w2, w1),
                  'CNOT %s %s' % (w1, w2)]
    return '\n'.join(basic_SWAP)
    
def preCompile(fileName):
    with open(fileName, 'r') as fN, open('basic_' + fileName, 'w+') as fNbasic:
        for fNline in fN:
            fNlist = fNline.strip().split()
            
            if fNlist[0] == 'NOT':      fNbasic.write(compileNOT(fNlist) + '\n')
            elif fNlist[0] == 'Rz':     fNbasic.write(compileRz(fNlist) + '\n')
            elif fNlist[0] == 'CRz':    fNbasic.write(compileCRz(fNlist) + '\n')
            elif fNlist[0] == 'CPHASE': fNbasic.write(compileCPhase(fNlist) + '\n')
            elif fNlist[0] == 'SWAP':   fNbasic.write(compileSWAP(fNlist) + '\n')
            else:                       fNbasic.write(' '.join(fNlist) + '\n')

# The code below test the non-atomic gates and the pre-compiling
'''
preCompile('non_atomic_gates.circuit')
num_wires_total, circuit_gate_list = ReadInput('basic_non_atomic_gates.circuit')
myInputState = GetInputState(num_wires_total, circuit_gate_list)
for gates in circuit_gate_list:
    if gates[0] == 'H':
        H_wire = int(gates[1])
        Hadamard(H_wire,myInputState)
    elif gates[0] == 'P':
        P_wire = int(gates[1])
        myPhase = float(gates[2])
        Phase(H_wire, myPhase, myInputState)
    elif gates[0] == 'CNOT':
        myW_1 = int(gates[1])
        myW_2 = int(gates[2])
        CNOT(myW_1, myW_2, myInputState)

print(myInputState)
'''
###############################################################################
############################### Phase Estimation ##############################
###############################################################################


