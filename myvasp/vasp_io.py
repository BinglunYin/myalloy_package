#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy




def my_read_vasp(filename):
    from ase.io.vasp import read_vasp 
    
    atoms = read_vasp(filename)
    with open(filename, 'r') as f:
        atoms.pos_a0 = float( f.readlines()[1] )

    cn = get_cn(atoms)
    atoms.cn = cn 
    return atoms




def get_cn(atoms):
    import pandas as pd

    natoms = atoms.get_positions().shape[0]
    atoms_an = atoms.get_atomic_numbers()

    # number of atoms of each element
    natoms_elem = np.array([])
    for i in pd.unique(atoms_an):
        mask = np.isin(atoms_an, i)
        natoms_elem = np.append( natoms_elem, \
            atoms_an[mask].shape[0] )

    if np.abs( natoms - natoms_elem.sum() ) > 1e-10:
        sys.exit('==> ABORT. wrong natoms_elem. ')
    cn = natoms_elem / natoms
    return cn




def my_write_vasp(atoms_in, filename='POSCAR', vasp5=True):
    from ase.io.vasp import write_vasp

    atoms = copy.deepcopy(atoms_in)
    pos_a0 = atoms.pos_a0

    atoms.set_cell( atoms.get_cell()/pos_a0 )
    atoms.set_positions( atoms.get_positions()/pos_a0, \
        apply_constraint=False ) 
    
    write_vasp('POSCAR_temp', atoms,
    label='system_name', direct=False, vasp5=vasp5)

    with open('POSCAR_temp', 'r') as f:
        lines = f.readlines()
    lines[1] = ' %.8f \n' % (pos_a0)

    with open(filename, "w") as f:
        f.writelines(lines)
    os.remove('POSCAR_temp')



def my_rm(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

