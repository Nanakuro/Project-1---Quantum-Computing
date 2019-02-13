#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:57:55 2019

@author: Minh Nguyen
"""

import numpy as np
from math import log, gcd
from random import randint

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
    for root in range(2,int(log(N,2))):
        if N**(1/root) == int(N**(1/root)):
            return root
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
    
def ShorClassic(N):
    if CheckForFactor(N):
        x = randint(2, int(N**(0.5)))
        x_list = []
        while len(x_list) <= (int(N**(0.5))-2)+1:
            x = randint(2, int(N**(0.5)))
            if x not in x_list:
                if gcd(x,N) != 1:
                    x_list.append(x)
                    x = randint(1,int(N**(0.5)))
                    continue
                r = PeriodClassic(x,N)
                if r != 0:
                    fact1 = gcd(x**(r//2) - 1, N)
                    fact2 = gcd(x**(r//2) + 1, N)
                    if fact1 == 1 or fact2 == N:
                        x_list.append(x)
                        continue
                    else:
                        print('x = %d, r = %d' % (x,r))
                        print('N = %d = %d x %d' % (N, fact1, fact2))
                        return (x, r, fact1, fact2)
            continue
    elif N % 2 == 0:
        print('N is even.')
        print('N = %d = %d x %d' % (N, 2, N/2))
        return (0,0,2, N/2)
    elif IntegerPowerRoots(N) != 0:
        N_root = IntegerPowerRoots(N)
        print('N has integer power roots.')
        print('N = %d = %d x %d' % (N, N**(1/N_root), N**(1-1/root)))
        return (0,0,2, N/2)
    print('N has no non-trivial factors.')
    print('N = %d = %d x %d' % (N, 1, N))
    return (0,0,1,N)

myFile = 'shor_factor_list.txt'

num_bits = 8
print(int('1'*num_bits,2))
number_list = [ n for n in range(int('1'*num_bits,2)) ]

#my_x, my_r, f1, f2 = ShorClassic(1999)


with open(myFile,'w+') as f:
    f.write('N \t F1 \t F2 \t x \t r\n')
    for my_N in number_list:
        if CheckForFactor(my_N):
            my_x, my_r, f1,f2 = ShorClassic(my_N)
            print(my_N,f1, f2, my_x, my_r)
            f.write('%d \t %d \t %d \t %d \t %d\n' % (my_N, f1, f2, my_x, my_r))
