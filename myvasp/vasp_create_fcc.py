#!/home/yin/opt/bin/python3

import numpy as np
from myvasp import vasp_func as vf



def vasp_create_fcc_111(a, ncell, bp=0):
    print('==> create fcc 111 plane: ')
    print(a, ncell, bp)

    latt = np.array([
        [1, 0, 0],
        [-0.5, np.sqrt(3)/2, 0],   
        [0, 0, np.sqrt(6)],
    ]) * a/np.sqrt(2)
   
    motif = np.array([
        [0, 0, 0],
        [2/3,  1/3,  1/3],
        [1/3,  2/3,  2/3],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)
    atoms.pos_a0 = a 

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1]) *a 
        atoms.wrap()

    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)




def vasp_create_fcc_111_ortho(a, ncell, bp=0):
    print('==> create fcc 111 plane (ortho): ')
    print(a, ncell, bp)

    latt = np.array([
        [1, 0, 0],
        [0, np.sqrt(3), 0],   
        [0, 0, np.sqrt(6)],
    ]) * a/np.sqrt(2)
   
    motif = np.array([
        [  0,   0,   0],
        [1/2, 1/2,   0],
        [  0, 4/6, 1/3],
        [1/2, 1/6, 1/3],
        [  0, 2/6, 2/3],
        [1/2, 5/6, 2/3],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)
    atoms.pos_a0 = a 

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1]) *a 
        atoms.wrap()

    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)




def vasp_create_fcc_100(a, ncell, bp=0):
    print('==> create fcc 100 plane: ')
    print(a, ncell, bp)

    latt = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],   
        [0, 0, 1.0],
    ]) * a
   
    motif = np.array([
        [0, 0, 0],
        [0.5,  0.5,  0],
        [0.5,  0,  0.5],
        [0,  0.5,  0.5],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)
    atoms.pos_a0 = a 

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1]) *a 
        atoms.wrap()

    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)






def vasp_create_fcc_100_min(a, ncell, bp=0):
    print('==> create fcc 100 plane min: ')
    print(a, ncell, bp)

    latt = np.array([
        [1.0/np.sqrt(2), 0, 0],
        [0, 1.0/np.sqrt(2), 0],   
        [0, 0, 1.0],
    ]) * a
   
    motif = np.array([
        [0, 0, 0],
        [0.5,  0.5,  0.5],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)
    atoms.pos_a0 = a 

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1]) *a 
        atoms.wrap()

    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)






def vasp_create_fcc_111_min(a, ncell, bp=0):
    print('==> create fcc 111 plane min: ')
    print(a, ncell, bp)

    latt = np.array([
        [1, 0, 0],
        [-0.5, np.sqrt(3)/2, 0],   
        [ 0.5, np.sqrt(3)/6, np.sqrt(6)/3],
    ]) * a/np.sqrt(2)
   
    motif = np.array([
        [0, 0, 0],
    ])

    atoms = vf.create_supercell(latt, motif, ncell)
    atoms.pos_a0 = a 

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1]) *a 
        atoms.wrap()

    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)

