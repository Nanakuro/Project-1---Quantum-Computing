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
        num = int(s[1],2)
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
            basis = "{0:b}".format(i)
            state.append([v,basis])
            l = len(basis)
    for s in state:
        s[1] = s[1].zfill(l)
    state = [tuple(s) for s in state]
    return state
            
        
    




testState = [
        (np.sqrt(0.1)*1.j, '101'),
        (np.sqrt(0.5), '000') ,
        (-np.sqrt(0.4), '010' )
        ]

#PrettyPrintBinary(testState)
#PrettyPrintInteger(testState)

print(StateToVec(testState))
print(VecToState(StateToVec(testState)))

