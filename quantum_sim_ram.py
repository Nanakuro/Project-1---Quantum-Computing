#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 02 02:05:33 2019

@author: Minh Nguyen
"""

import numpy as np
from matplotlib import pyplot as plt

file_name = 'ram_usage.txt'
with open(file_name,'r') as f:
    n_wires_list = f.readline().strip().split()[1:]
    Ia_ram = f.readline().strip().split()[1:]
    Ib_ram = f.readline().strip().split()[1:]
    Ic_ram = f.readline().strip().split()[1:]
    
    print(len(n_wires_list), len(Ia_ram), len(Ib_ram), len(Ic_ram))

Ia_ram = [ float(a) for a in Ia_ram ]
Ib_ram = [ float(b) for b in Ib_ram ]
Ic_ram = [ float(c) for c in Ic_ram ]

fig = plt.figure()
plt.xlabel('Number of wires')
plt.ylabel('Ram usage (MB)')
plt.title('Ia vs Ib vs Ic RAM')
plt.plot(n_wires_list[:-1],Ia_ram,'-o', label='Ia')
plt.plot(n_wires_list[:-1],Ib_ram,'-o', label='Ib')
plt.plot(n_wires_list,Ic_ram,'-o', label='Ic')
plt.savefig('ram_comparison.png')
