#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:09:53 2019

@author: Minh Nguyen
"""

import numpy as np
from random import random, randint, choices

def RandomPNOTCircuit(fileName):
    with open(fileName, 'w+') as f:
        num_wires = randint(1,4)
        f.write('%d\n' % num_wires)
        
        f.write('INITSTATE BASIS |%s>\n' % bin(randint(1,2**num_wires))[2:])
        
        gate_list = choices(['P','NOT'],k=randint(5,20))
        for gate in gate_list:
            wire = randint(0,num_wires-1)
            if gate == 'P':
                phase = random()*2*np.pi
                f.write('P %d %.9f\n' % (wire, phase))
            elif gate == 'NOT':
                f.write('NOT %d\n' % wire)

RandomPNOTCircuit('my_random.circuit')