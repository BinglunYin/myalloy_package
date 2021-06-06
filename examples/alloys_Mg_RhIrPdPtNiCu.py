
import numpy as np
from myalloy import main 


# Mg 

a1 = main.alloy_class(name='Mg', \
    cn=np.array([1]), brav_latt='hcp')

a1.a = 3.19023441
a1.c = 1.62723943 * a1.a 

a1.Cij = np.array([ 61, 28, 22, 64, 18 ])

a1.gamma = 166

a1.calc_stroh(slip_system='pyr2_ca_screw', \
    param={
        'bp':np.array([0, 1]),    # compact core
        'r12':60, 
        'output_name':'Mg',
        })








# some results in 
# https://www.nature.com/articles/s41524-019-0151-x 


a2 = main.alloy_class(name='RhIrPdPtNiCu', \
    cn=np.array([1, 1, 1, 1, 1, 1]))

a2.a = 3.811
a2.dV = np.array([ 0.253,  0.767,  \
                   1.412,  1.835,  \
                  -2.581, -1.686   ])

C11=np.mean([286, 296, 297,  284, 288, 286])
C12=np.mean([176, 176, 182,  172, 171, 176])
C44=np.mean([111, 113, 113,  111, 110, 112])

a2.Cij = np.array([ C11, C12, C44 ])

a2.calc_yield_strength({'filename':'RhIrPdPtNiCu', \
    'alpha':0.123, 'et':1e-4, 'A':1})


a2.gamma = 138
pos_in = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0], 
    ])
a2.calc_stroh(slip_system='111_a_edge', \
    param={
        'pos_in':pos_in, 
        'output_name':'RhIrPdPtNiCu',
        })






