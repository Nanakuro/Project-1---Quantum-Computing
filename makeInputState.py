import sys
import numpy as np
from numpy import random as rd

def normalVector(dim):
    vector = [ rd.normal(0,1)+rd.normal(0,1)*1.j for i in range(dim) ]
    magnitude = np.sqrt(sum(np.abs(v)**2 for v in vector))
    vector = [ v/magnitude for v in vector ]
    return vector

n = int(sys.argv[1])
C_vector = normalVector(2**n)
with open('myInputState%d.dms' % n,'w+') as f:
    for vec in C_vector:
        f.write('%.9f %.9f\n' %(vec.real, vec.imag))
