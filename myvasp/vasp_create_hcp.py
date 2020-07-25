#!/home/yin/opt/bin/python3

import numpy as np
from myvasp import vasp_func as vf


def vasp_create_hcp_basal(a, ca, ncell, bp=33):
    print('==> create hcp basal plane: ')
    print(a, ca, ncell, bp)

    latt = np.array([
        [1.0, 0, 0],
        [-0.5, np.sqrt(3)/2, 0],   
        [0, 0, ca],
    ]) * a
    
    motif = np.array([
        [0, 0, 0],
        [1/3,  2/3,  1/2],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1])*a
        atoms.wrap()

    atoms.pos_a0 = a 
    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)






def vasp_create_hcp_basal_ortho(a, ca, ncell, bp=33):
    print('==> create hcp basal_ortho plane: ')
    print(a, ca, ncell, bp)

    latt = np.array([
        [ 1.0,          0,  0],
        [   0, np.sqrt(3),  0],   
        [   0,          0, ca],
    ]) * a
    
    motif = np.array([
        [0.0,    0,    0],
        [1/2,  1/2,    0],
        [  0,  2/6,  1/2],
        [1/2,  5/6,  1/2],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)
    
    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1])*a
        atoms.wrap()

    atoms.pos_a0 = a 
    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)








def vasp_create_hcp_prism1(a, ca, ncell, bp=33):
    print('==> create hcp prism plane: ')
    print(a, ca, ncell, bp)

    ncell2 = ncell.copy()
    for i in np.arange(3):
        ncell2[i] = ncell[ np.mod(i+2, 3) ]
    
    vasp_create_hcp_basal(a, ca, ncell2, bp=0)

    atoms = vf.my_read_vasp('POSCAR')
    atoms = vf.make_SFP_xy(atoms, i1=1)
    atoms = vf.make_a3_ortho(atoms)

    if bp == 33:
        print('==> create prism1-W')
        atoms.positions = atoms.positions + np.array([0, 0, 0.1])*a
        atoms.wrap()
    
    elif bp == -33:
        print('==> create prism1-N')
        atoms.positions = atoms.positions + np.array([0, 0, -0.1])*a
        atoms.wrap()
    
    atoms.pos_a0 = a 
    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)








def vasp_create_hcp_pyr1(a, ca, ncell, bp=33):
    print('==> create hcp basal pyr1: ')
    print(a, ca, ncell, bp)

    latt = np.array([
        [ 1.0, 0, 0],
        [ 0.5, np.sqrt(3)/2, 0],   
        [-1.0, 0, ca],
    ]) * a
    
    motif = np.array([
        [0, 0, 0],
        [1/6, 2/3, 1/2],
    ])

    ncell2 = ncell.copy()
    for i in np.arange(3):
        ncell2[i] = ncell[ np.mod(i+2, 3) ]
   
    atoms = vf.create_supercell(latt, motif, ncell2)
    atoms = vf.make_SFP_xy(atoms, i1=1)
    atoms = vf.make_a3_ortho(atoms)

    if bp == 33:
        print('==> create pry1-W')
        atoms.positions = atoms.positions + np.array([0, 0, 0.1])*a
        atoms.wrap()

    elif bp == -33:
        print('==> create pry1-N')
        atoms.positions = atoms.positions + np.array([0, 0, -0.05])*a
        atoms.wrap()
    
        
    atoms.pos_a0 = a 
    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)






def vasp_create_hcp_pyr2(a, ca, ncell, bp=33):
    print('==> create hcp basal pyr2: ')
    print(a, ca, ncell, bp)

    latt = np.array([
        [ 1.0, 0, 0],
        [ 0.0, np.sqrt(3), 0],   
        [-1.0, 0, ca],
    ]) * a
    
    motif = np.array([
        [  0,    0,    0],
        [1/2,  1/2,    0],
        [  0,  5/6,  1/2],
        [1/2,  1/3,  1/2]
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

# a = 3.23415
# ca = 1.5992

# vasp_create_hcp_basal(a, ca, np.array([1, 1, 10]) )
# vasp_create_hcp_prism1(a, ca, np.array([1, 1, 16]) )
# vasp_create_hcp_pyr1(a, ca, np.array([1, 1, 16]), bp=-33)
# vasp_create_hcp_pyr1(a, ca, np.array([1, 1, 16]), bp=33)
# vasp_create_hcp_pyr2(a, ca, np.array([1, 1, 10]) )



