#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy
from myvasp import vasp_func as vf 




def get_list_of_outcar():
    from ase.io.vasp import read_vasp_out

    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    latoms2 = []   # list of ASE_Atoms from OUTCAR
    for i in jobn:
        filename = './y_dir/%s/OUTCAR' %(i)
        atoms2 = read_vasp_out(filename)
        latoms2.append(atoms2)
    return latoms2




def get_list_of_atoms():
    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    latoms = []   # list of ASE_Atoms from CONTCAR
    
    os.chdir('y_dir')
    for i in np.arange( len(jobn) ):
        os.chdir( jobn[i] )
        atoms = my_read_vasp('CONTCAR')
        latoms.append(atoms)
        os.chdir('..')
    os.chdir('..')

    return latoms




#==============================


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

