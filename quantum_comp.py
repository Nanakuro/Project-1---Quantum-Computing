#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:09:53 2019

@author: Minh Nguyen
"""

import numpy as np

def PrettyPrintBinary(state):
    i = 1
    for s in state:
        if i == len(state):
            print('{:g} |{:s}>'.format(s[0], s[1]))
        else:
            print('{:g} |{:s}>'.format(s[0], s[1]),end=' + ')
        i += 1

def PrettyPrintInteger(state):
    i = 1
    for s in state:
        num = int(s[1],2)   # Convert binary string to base-10 integer
        if i == len(state):
            print('{:g} |{:g}>'.format(s[0], num))
        else:
            print('{:g} |{:g}>'.format(s[0], num),end=' + ')
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
         
##############################

   
def tensorProd(matrix_list):
    product = matrix_list[0]
    i = 1
    while i < len(matrix_list):
        product = np.kron(matrix_list[i],product)
        i += 1
    return product

def HadamardArray(i,k):     # Apply Hadamard matrix to wire i out of k wires
    Hadamard = 1/np.sqrt(2) * np.array([[1,1],
                                        [1,-1]])
    M_list = []
    for j in range(k):
        if j == k-i-1:
            M_list.append(Hadamard)
            continue
        M_list.append(np.identity(2))
    
    matrix = tensorProd(M_list)
    print(matrix)
    return matrix

def PhaseArray(i,k,phi):
    Phase = np.array([[1,           0],
                      [0,   np.exp(phi*1.j)]])
    M_list = []
    for w in range(k):
        if w == k-i-1:
            M_list.append(Phase)
        else:
            M_list.append(np.identity(2))
    
    matrix = tensorProd(M_list)
    print(matrix)
    return matrix

# Currently only apply to target wires that is next to a control wire.
def CNOTArray(control,target,total):
    if total < 2:
        print('Not enough number of qubits to implement CNOT gate.')
    elif control == target:
        print('Invalid placement of CNOT gate (control wire and target wire must be different)')
    else:
        CNOTdown = np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,0,1],
                         [0,0,1,0]])    
    
        CNOTup = np.array([[1,0,0,0],
                           [0,0,0,1],
                           [0,0,1,0],
                           [0,1,0,0]])
        
        M_list = []
        w = 0
        while w < total:
            if target < control and w == total-1-control:
                M_list.append(CNOTup)
                w += 1
            elif target > control and w == total-1-target:
                M_list.append(CNOTdown)
                w += 1
            else:
                M_list.append(np.identity(2))
            w += 1
        matrix = tensorProd(M_list)
        #print(matrix)
        return matrix
                
def ReadInput(fileName):
        myInput_lines=open(fileName).readlines()
        myInput=[]
        numberOfWires=int(myInput_lines[0])
        for line in myInput_lines[1:]:
            myInput.append(line.split())
        return (numberOfWires,myInput)
    
    




#testState = [
#        (np.sqrt(0.1)*1.j, '101'),
#        (np.sqrt(0.5), '000') ,
#        (-np.sqrt(0.4), '010' )
#        ]
#testState2 = [
#        (np.sqrt(0.1)*3.j+2, '111'),
#        (np.sqrt(0.5)-0.1, '110') ,
#        (-np.sqrt(0.4)-2.j, '001' )
#        ]

#PrettyPrintBinary(testState2)
#PrettyPrintInteger(testState2)

#print(StateToVec(testState2))
#print(VecToState(StateToVec(testState2)))

#HadamardArray(1,2)
#PhaseArray(0,2,0.1)
dim = 3
testState = [(i,'{0:b}'.format(i).zfill(dim)) for i in range(2**dim)]
