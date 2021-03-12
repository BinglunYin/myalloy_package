#!/home/yin/opt/bin/python3

import numpy as np
from myvasp import vasp_func as vf



def vasp_create_bcc_110(a, ncell, bp=0):
    print('==> create bcc 110 plane: ')
    print(a, ncell, bp)

    latt = np.array([
        [ 1.0,     0,     0],
        [ 1/2,  -1/2,   1/2],
        [ 1/2,  -1/2,  -1/2],
    ]) *a

   
    motif = np.array([
        [0, 0, 0],
    ])

    ncell2 = ncell.copy()
    for i in np.arange(3):
        ncell2[i] = ncell[ np.mod(i+2, 3) ]
   
    atoms = vf.create_supercell(latt, motif, ncell2)
    atoms = vf.make_SFP_xy(atoms, i1=1)
    atoms = vf.make_a3_ortho(atoms)


    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1])*a
        atoms.wrap()

    atoms.pos_a0 = a 
    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)





#===================
# examples:

# a = 3.308

# vasp_create_bcc_110(a, np.array([1, 1, 6]), bp=33 )




