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





def make_SFP_xy(atoms_in, i1):
    import numpy.matlib

    atoms = atoms_in.copy()
    pos = atoms.get_positions()
    latt = atoms.get_cell()[:]
    
    # i1 = 0, 1, 2. The axis to be new a1.
    i2 = np.mod(i1+1, 3)
    print('rotate SFP (lattices %.0f, %.0f) to xy' %(i1, i2) )

    # old coordinate basis
    ox=np.array([
        [1.0,   0,   0],
        [  0, 1.0,   0],
        [  0,   0, 1.0]  ])

    # new coordinate basis
    nx = np.zeros([3, 3])
    nx[0,:] = latt[i1,:] / np.linalg.norm( latt[i1,:] )
    
    temp = latt[i2,:] - np.dot(latt[i2,:], nx[0,:]) * nx[0,:]
    nx[1,:] = temp / np.linalg.norm(temp)
    
    temp = np.cross(nx[0,:], nx[1,:])
    nx[2,:] = temp / np.linalg.norm(temp)
    print('nx:', nx)

    # v_old @ ox = v_new @ nx 
    R = ox @ np.linalg.inv(nx)
    print('R:', R )

    temp = numpy.matlib.repmat( latt @ R, 2, 1)
    newlatt = np.zeros([3, 3])
    for i in np.arange(3):
        newlatt[i,:] = temp[i1+i,:]
    
    newpos = pos @ R
    
    atoms2 = atoms.copy()
    atoms2.set_positions(newpos, apply_constraint=False)
    atoms2.set_cell(newlatt)
    atoms2.wrap()
    return atoms2
    




def make_a3_ortho(atoms_in):
    atoms = atoms_in.copy()
    latt = atoms.get_cell()[:]
    
    if np.abs(latt[0,1]) < 1e-10 \
        and  np.abs(latt[0,2]) < 1e-10 \
        and  np.abs(latt[1,2]) < 1e-10 :
        
        k = np.round( latt[2,1] /  latt[1,1] )
        latt[2,:] = latt[2,:] - k* latt[1,:] 
    
        k = np.round( latt[2,0] /  latt[0,0] )
        latt[2,:] = latt[2,:] - k* latt[0,:] 
    
        atoms.set_cell( latt )
        atoms.wrap()
    
    return atoms 
    




def create_random_alloys(atoms_in, cn, nsamples=1, filename='POSCAR', id1=1, vasp5=False):
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
  
        filename2 = '%s_%03d' %( filename, i+id1 )
        vf.my_write_vasp(atoms, filename2, vasp5=vasp5)








def vasp_create_twin(atoms_in):
    from myvasp import vasp_shift_to_complete_layers as vfs

    atoms = copy.deepcopy(atoms_in)

    latt   = atoms.get_cell()[:]
    vf.confirm_0( latt[2, 0:2] )

    pos = atoms.get_positions()
    an  = atoms.get_atomic_numbers()
    data = np.hstack([ pos, an[:, np.newaxis ] ])
   
    mask = np.argsort( data[:,2] )   # by z
    data = data[mask,:]
    
    natoms = pos.shape[0]
    nlayers, nmiss = vfs.check_layers(atoms) 
    vf.confirm_int( natoms/nlayers )
    vf.confirm_int( natoms/2 ) 

    for i in np.arange( int(natoms/nlayers), int(natoms/2) ):
        data[natoms-i, 0:2] = data[i, 0:2].copy()  
        data[natoms-i,   3] = data[i,   3].copy()   
    
    mask = np.argsort( data[:,3] )
    data = data[mask,:]
    
    pos_a0 = atoms.pos_a0
    lelem  = data[:, 3]
    pos    = data[:, 0:3]
    
    vf.write_poscar(pos_a0, latt, lelem, pos, filename='POSCAR_twin')

