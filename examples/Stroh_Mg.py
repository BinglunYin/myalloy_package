
import numpy as np
from myalloy import main 


# Mg MEAM
# https://doi.org/10.1016/j.actamat.2016.08.002




a1 = main.alloy_class(name='Mg', \
    cn=np.array([1]), brav_latt='hcp')

a1.a = 3.187
a1.c = 1.623*a1.a 

a1.Cij = np.array([ 64.27, 25.45, 20.29, 70.93, 18.02 ])





# c+a cross-slip

a1.gamma = 169 

a1.calc_stroh(slip_system='pyr1_ca_screw', \
    param={
        'bp': np.array([0, 0.422]),
        'output_name': 'Mg_pyr1_ca',
        })



a1.gamma = 200 

a1.calc_stroh(slip_system='pyr2_ca_screw', \
    param={
        'bp': np.array([0, 0.479]),
        'output_name': 'Mg_pyr2_ca',
        })






# <a> cross slip 

a1.gamma = 23

a1.calc_stroh(slip_system='basal_a_screw', \
    param={
        'output_name': 'Mg_basal_a',
        })



a1.gamma = 219 

a1.calc_stroh(slip_system='prism_a_screw', \
    param={
        'bp': np.array([0.5, 0]),
        'output_name': 'Mg_prism_a',
        })



a1.gamma = 171

a1.calc_stroh(slip_system='pyr1_a_screw', \
    param={
        'bp': np.array([0.5, -0.116]),
        'output_name': 'Mg_pyr1_a',
        })



