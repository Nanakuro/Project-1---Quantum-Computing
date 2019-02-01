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
    state = [tuple(s) for s in state]
    return state
         
############################## Quantum Simulator Ia ###########################

'''   
def tensorProd(matrix_list):
    product = matrix_list[0]
    i = 1
    while i < len(matrix_list):
        product = np.kron(matrix_list[i],product)
        i += 1
    return product
'''
def HadamardArray(wire,total):     # Apply Hadamard matrix to wire i out of k wires
    Hadamard = 1/np.sqrt(2) * np.array([[1,1],
                                        [1,-1]])
    matrix = np.identity(2)
    if wire == 0:
        matrix = Hadamard
    w = 1
    #M_list = []
    while w < total:
        if w == total-wire-1:
            matrix = np.kron(Hadamard, matrix)
            #M_list.append(Hadamard)
        else:
            matrix = np.kron(np.identity(2), matrix)
            #M_list.append(np.identity(2))
        w += 1
    
    #matrix = tensorProd(M_list)
    return matrix

def PhaseArray(wire,total,phi):
    Phase = np.array([[1,           0],
                      [0,   np.exp(phi*1.j)]])
    #M_list = []
    matrix = np.identity(2)
    if wire == 0: matrix = Phase
    w = 1
    while w < total:
        if w == total-wire-1:
            matrix = np.kron(Phase, matrix)
            #M_list.append(Phase)
        else:
            matrix = np.kron(np.identity(2), matrix)
            #M_list.append(np.identity(2))
        w += 1
    
    #matrix = tensorProd(M_list)
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
        if control == 0:
            matrix = CNOTup
            w += 1
        elif target == 0:
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
            state = [(1,input_circuit[0][2].strip('|').strip('>'))]
            return state
    else:
        return [(1,'0'*numberOfWires)]
'''
############################ Quantum Simulator Ib #############################

def computeState(vector, matrix_list):
    if len(matrix_list) == 1:
        r = matrix_list[0] @ vector
        return r
    else:
        v = matrix_list[0] @ vector
        m = matrix_list[1:]
        return computeState(v,m)
'''
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



