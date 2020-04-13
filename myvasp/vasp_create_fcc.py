#!/home/yin/opt/bin/python3

import numpy as np
from ase.io.vasp import write_vasp
from ase import Atoms
import os


def vasp_create_fcc_111(ncell, bp, a0=1.0):
    print('==> create fcc 111 plane: ')
    print(ncell, bp, a0)

    latt = np.array([
        [1, 0, 0],
        [-0.5, np.sqrt(3)/2, 0],   
        [0, 0, np.sqrt(6)],
    ]) * a0/np.sqrt(2)
   
    motif = np.array([
        [0, 0, 0],
        [2/3,  1/3,  1/3],
        [1/3,  2/3,  2/3],
    ])

    atoms = create_supercell(ncell, latt, motif)

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.01])
        atoms.wrap()

    write_poscar_with_a0(atoms, a0)






def vasp_create_fcc_100(ncell, bp, a0=1.0):
    print('==> create fcc 100 plane: ')
    print(ncell, bp, a0)

    latt = np.array([
        [1.0, 0, 0],
        [0, 1.0, 0],   
        [0, 0, 1.0],
    ]) * a0
   
    motif = np.array([
        [0, 0, 0],
        [0.5,  0.5,  0],
        [0.5,  0,  0.5],
        [0,  0.5,  0.5],
    ])

    atoms = create_supercell(ncell, latt, motif)

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.01])
        atoms.wrap()

    write_poscar_with_a0(atoms, a0)




def create_supercell(ncell, latt, motif):
    atoms_pos = np.zeros([1, 3])
    
    for k in np.arange(ncell[2]):
        for j in np.arange(ncell[1]):
            for i in np.arange(ncell[0]):
                
                refp = i*latt[0,:] + j*latt[1,:] + k*latt[2,:] 

                for m in np.arange(motif.shape[0]):
                    atoms_pos = np.vstack([ atoms_pos, refp +motif[m,:]@latt ])
               
    atoms_pos = np.delete(atoms_pos, 0, 0)       
       
    superlatt = ncell @ latt

    atoms = Atoms(cell = superlatt,
        positions = atoms_pos, 
        pbc = [1, 1, 1],
        )
    
    natoms = atoms.positions.shape[0]
    atoms.set_chemical_symbols( np.ones([natoms, 1]) )

    return atoms



def write_poscar_with_a0(atoms, a0, filename='POSCAR', vasp5=True):
     
    atoms.set_cell( atoms.cell[:]/a0 )
    atoms.set_positions( atoms.positions/a0, apply_constraint=False ) 
    
    write_vasp('POSCAR_temp', atoms,
    label='system_name', direct=False, vasp5=vasp5)

    with open('POSCAR_temp') as f:
        lines = f.readlines()
  
    lines[1] = ' %.8f \n' % (a0)

    with open(filename, "w") as f:
        f.writelines(lines)

    os.remove('POSCAR_temp')

    # why to add this?
#    atoms.set_cell( atoms.cell[:] *a0 )
#    atoms.set_positions( atoms.positions *a0, apply_constraint=False ) 
    



