#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy
from myvasp import vasp_func as vf 






def bestsqs_to_POSCAR(filename='bestsqs-1.out'):

    with open(filename) as f:
        sqs = f.read().splitlines()


    latt = np.zeros([3, 3])
    for i in np.arange(3):
        temp = sqs[3+i].split(' ')

        for j in np.arange(3):
            latt[i, j] = float( temp[j] )


    natoms = len(sqs)-6
    pos = np.zeros([natoms, 3])
    lelem = np.zeros( natoms )

    for i in np.arange(natoms):
        temp = sqs[6+i].split(' ')

        for j in np.arange(3):
            pos[i, j] = float( temp[j] )
    
        s = temp[3]
        if s == 'A':
            t = 1
        elif s == 'B':
            t = 2
        elif s == 'C':
            t = 3
        elif s == 'D':
            t = 4
        elif s == 'E':
            t = 5
        elif s == 'F':
            t = 6
        elif s == 'G':
            t = 7
        lelem[i] = t 
    
    temp = lelem[:].argsort()
    pos   =   pos[ temp, :]
    lelem = lelem[ temp   ]

    temp = 'POSCAR_'+filename[8]
    a_pos = 4.0
    write_poscar(a_pos, latt*a_pos, lelem, pos*a_pos, filename=temp)







def write_poscar(a_pos, latt, lelem, pos, filename='POSCAR'):

    latt = latt/a_pos 
    pos  = pos/a_pos 

    temp = lelem.max()
    ns = np.zeros(temp) 

    for i in np.arange( len(ns) ):
        mask = lelem[:]==(i+1) 
        temp = lelem[mask]
        ns[i] = temp.shape[0]

    mask = ns[:] != 0
    ns = ns[mask]

    f = open(filename, 'w+')
    f.write('system name \n %22.16f \n' %(a_pos) )

    for i in np.arange(3):
        f.write(' %22.16f %22.16f %22.16f \n' %(latt[i,0], latt[i,1], latt[i,2])  )

    for i in np.arange(len(ns)):
        f.write(' %d ' %(ns[i]) )
    f.write('\nS \nC \n')

    for i in np.arange(pos.shape[0]):
        f.write(' %22.16f %22.16f %22.16f   T T T \n' %(pos[i,0], pos[i,1], pos[i,2])  )

    f.close() 

   






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
    import types 
    
    atoms = read_vasp(filename)
    with open(filename, 'r') as f:
        atoms.pos_a0 = float( f.readlines()[1] )

    atoms.get_cn    = types.MethodType(get_cn,    atoms) 
    atoms.get_nelem = types.MethodType(get_nelem, atoms) 

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





def get_nelem(atoms):
    import pandas as pd
    atoms_an = atoms.get_atomic_numbers()
    nelem = len( pd.unique(atoms_an) ) 
    return nelem 
    








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
    lines[1] = ' %.16f \n' % (pos_a0)

    with open(filename, "w") as f:
        f.writelines(lines)
    os.remove('POSCAR_temp')


