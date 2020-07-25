#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy
from myvasp import vasp_func as vf 




def create_supercell(latt, motif, ncell):
    from ase import Atoms

    atoms_pos = np.zeros([1, 3])
    
    for k in np.arange(ncell[2]):
        for j in np.arange(ncell[1]):
            for i in np.arange(ncell[0]):
                
                refp = i*latt[0,:] + j*latt[1,:] + k*latt[2,:] 

                for m in np.arange(motif.shape[0]):
                    atoms_pos = np.vstack([ atoms_pos, refp + motif[m,:] @ latt ])
               
    atoms_pos = np.delete(atoms_pos, 0, 0)       
       
    superlatt = latt.copy()
    for i in np.arange(3):
        superlatt[i,:] = superlatt[i,:] * ncell[i] 

    atoms = Atoms(cell = superlatt, positions = atoms_pos, 
        pbc = [1, 1, 1] )
    
    natoms = atoms.positions.shape[0]
    atoms.set_chemical_symbols( np.ones([natoms, 1]) )

    return atoms





def create_random_alloys(atoms_in, cn, nsamples=1, id1=1, vasp5=False):
    atoms = copy.deepcopy(atoms_in)
    natoms = atoms.get_positions().shape[0]
   
    # calc natoms_elem
    cn = cn/cn.sum()     
    natoms_elem = np.around( cn * natoms )

    if natoms_elem.min() < 0.1:
        sys.exit('==> ABORT. natoms is too small. ')

    max_elem = np.argmax( natoms_elem )
    temp = natoms_elem.sum() - natoms_elem[max_elem]
    natoms_elem[max_elem] = natoms - temp
    print(natoms_elem)

    # new symbols
    symb = np.array([])
    for i in np.arange(natoms_elem.shape[0]):
        for j in np.arange(natoms_elem[i]):
            symb = np.append(symb, i+1)
    atoms.set_chemical_symbols( symb )

    # randomize pos
    for i in np.arange(nsamples):
        temp = np.hstack([ atoms.positions, \
            np.random.random_sample([natoms, 1]) ])
        ind = np.argsort(temp[:, -1])
        atoms.set_positions(temp[ind, 0:3], apply_constraint=False)
  
        filename = 'POSCAR_%03d' %( i + id1 )
        vf.my_write_vasp(atoms, filename, vasp5=vasp5)




