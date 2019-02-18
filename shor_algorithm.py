#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:57:55 2019

@author: Minh Nguyen
"""
import numpy as np
from numpy.linalg import eigvals
from fractions import Fraction
#import phase_estimation as pe
import scipy.sparse as sp
from math import log, log2, gcd, ceil, pi
from random import choices
from cmath import phase
import quantum_sim_module as qs
import time

'''
https://stackoverflow.com/questions/1801391/what-is-the-best-algorithm-for-checking-if-a-number-is-prime
'''
def isPrime(n):
    if n <= 3 and isinstance(n,int):
        return n > 1
    elif n % 2 == 0 or n % 3 == 0:
        return False
    factor = 5
    step = 2
    
    while factor**2 <= n:
        if n % factor == 0:
            return False
        factor += step
        step = 6 - step
    
    return True

def IntegerPowerRoots(N):
    for a in range(2,int(log(N,2))):
        if abs(N**(1/a) - int(round(N**(1/a)))) < 10E-6:
            return a
    return 0
        
def CheckForFactor(N):
    return not isPrime(N) and (N % 2 != 0) and (IntegerPowerRoots(N) == 0) and N > 1

def PeriodClassic(x,N):
    r_power = 2
    while r_power > 0:
        if (x**r_power) % N == 1:
            if r_power % 2 == 0:
                return r_power
            else:
                return 0
        r_power += 1

def ShorAlgorithm(N, flag='c'):
    if CheckForFactor(N):
        x_list = [i for i in range(2,int(ceil(N**(0.5)))) ]
        while len(x_list) > 0:
            x = choices(x_list)[0]
            if gcd(x,N) != 1:
                #fact1 = gcd(x,N)
                #fact2 = N // fact1
                #print('%d = %d x %d' % (N, fact1, fact2))
                #return (x, 0, fact1, fact2)
                x_list.remove(x)
                continue

            if flag == 'c':
                r = PeriodClassic(x,N)
            elif flag == 'q':
                r = PeriodQuantumPE(x,N)
            elif flag == 'qM':
                r = PeriodQuantumMatrix(x,N)
            else:
                print('INVALID FLAG')
                raise KeyError

            if r != 0:
                fact1 = gcd((x**(r//2) - 1) % N, N)
                fact2 = gcd((x**(r//2) + 1) % N, N)
                if fact1 == 1 or fact2 == 1 or fact1 == N or fact2 == N:
                    continue
                #print('x = %d, r = %d' % (x,r))
                print('%d = %d x %d, x=%d r=%d' % (N, fact1, fact2, x, r))
                return (x, r, fact1, fact2)
            #x_list.remove(x)

    elif N % 2 == 0:
        print('N is even.')
        print('N = %d = %d x %d' % (N, 2, N/2))
        return (1, 0, 2, N/2)
    
    elif IntegerPowerRoots(N) != 0:
        N_root = IntegerPowerRoots(N)
        print('N has integer power roots.')
        print('N = %d = %d^%d' % (N, int(round(N**(1/N_root))), int(round(N**(1-1/N_root)))))
        return (1, 0, int(round(N**(1/N_root))), int(round(N**(1/N_root))))
    
    print('N has no non-trivial factors.')
    print('N = %d = %d x %d' % (N, 1, N))
    return (1, 0, 1, N)


def UnitaryM(x,N):
    n_qubits = int(ceil(log2(N)))
    uni_matrix = []
    for j_state in range(2**n_qubits):
        j_basis = [1.0, bin(j_state)[2:].zfill(n_qubits)]
        qs.stateModOp(x,N,j_basis)
        uni_matrix.append(qs.StateToVec([j_basis]))
    uni_matrix = sp.csr_matrix(uni_matrix).transpose()
    return uni_matrix
    
def PeriodQuantumMatrix(x,N):
    U_Matrix = UnitaryM(x,N)
    e_vals = eigvals(U_Matrix.todense()[:N,:N])
    e_vals = list(set(approximateList(e_vals)))
    rand_e_val = choices(e_vals)[0]
    theta = phase(rand_e_val)/(2*pi)
    if theta == 0:
        return 0
    r_period = Fraction(theta).limit_denominator(N).denominator
    if r_period % 2 != 0 or x**(r_period) % N != 1:
        return 0
    return r_period

def approximateList(num_list):
    new_list = []
    for n in num_list:
        re, im = n.real, n.imag
        if abs(round(re,9) - re) < 10E-9:
            re = round(re,9)
        if abs(round(im,9) - im) < 10E-9:
            im = round(im,9)
        new_list.append(complex(re,im))
    return new_list

def writeShorList(num_bits, mode):
    if mode == 'c':
        myFile = 'shor_factor_list.txt'
    elif mode == 'q':
        myFile = 'shor_factor_list_quantum.txt'
    else:
        print('INVALID MODE')
        raise KeyError
    number_list = [ n for n in range(int('1'*num_bits,2)+1) ]
    with open(myFile,'w+') as f:
        f.write('N = F1 x F2 \t x \t r\n')
        for my_N in number_list:
            if CheckForFactor(my_N):
                if mode == 'c':
                    my_x, my_r, f1,f2 = ShorAlgorithm(my_N)
                elif mode == 'q':
                    my_x, my_r, f1,f2 = ShorAlgorithm(my_N,flag='q')
                print('%d = %d x %d \t x=%d \t r=%d' % (my_N, f1, f2, my_x, my_r))
                #f.write('%d = %d x %d \t %d \t %d\n' % (my_N, f1, f2, my_x, my_r))
                #f.write('%d \t %d \t %d \t %d \t %d\n' % (my_N, my_x, my_r, f1, f2))     

def PeriodQuantumPE(x,N):
    bot = int(ceil(log2(N)))
    top = 2*bot+1
    init_state = [[1.0,'0'*top + '0'*(bot-1) + '1']]
    state = qs.PhaseEstimation(top,'CFUNC', init_state, x_val=x, N_val=N)
    #print(state)
    #theta = qs.GetMaxTheta(top,state)
    measured_state = qs.Measure(state)
    theta = sum([int(s)/2**(i+1) for i,s in enumerate(measured_state[:top])])
    print(theta)
    if theta == 0:
        return 0
    r_period = Fraction(theta).limit_denominator(N).denominator
    print(r_period)
    if r_period % 2 != 0 or x**(r_period) % N != 1:
        return 0
    return r_period

#writeShorList(6,'c')

start_time = time.time()
x,r,f1,f2 = ShorAlgorithm(21,flag='q')
run_time = time.time() - start_time
print('Run time is %fs' % run_time)

'''
x, N = 5, 33
myMatrix = UnitaryM(x,N)
e_vec = eigvals(myMatrix.todense()[:N,:N])
e_vec = list(set(approximateList(e_vec)))
theta_list = np.array([ phase(e)/(2*pi) for e in e_vec ])
rand_theta = choices(theta_list)[0]
while rand_theta == 0:
    rand_theta = choices(theta_list)[0]
r = Fraction(rand_theta).limit_denominator(N).denominator
while r % 2 != 0 or x**r % N != 1:
    rand_theta = choices(theta_list)[0]
    while rand_theta == 0:
        rand_theta = choices(theta_list)[0]
    r = Fraction(rand_theta).limit_denominator(N).denominator

print('x=%d \t N=%d' % (x,N))
print('random eigenvalue = ', np.exp(2j*pi*rand_theta))
print('theta = ', rand_theta)
print('r = ', r)
'''
#print('e =', theta_list)
#print('e*r =', (theta_list * r))
