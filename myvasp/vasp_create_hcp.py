#!/home/yin/opt/bin/python3

import numpy as np
import numpy.matlib
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
    atoms = make_SFP_xy(atoms, i1=1)
    atoms = make_a3_ortho(atoms)

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
    atoms = make_SFP_xy(atoms, i1=1)
    atoms = make_a3_ortho(atoms)

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
    atoms = make_SFP_xy(atoms, i1=1)
    atoms = make_a3_ortho(atoms)
    

    if bp == 33:
        atoms.positions = atoms.positions + np.array([0, 0, 0.1])*a
        atoms.wrap()

    atoms.pos_a0 = a 
    vf.my_write_vasp(atoms, filename='POSCAR', vasp5=True)





#=======================


def make_SFP_xy(atoms_in, i1):
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
    
    temp=latt[i2,:] - np.dot(latt[i2,:], nx[0,:]) * nx[0,:]
    nx[1,:]=temp / np.linalg.norm(temp)
    
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
    atoms2.set_positions(newpos)
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
    





#===================
# examples:

# a = 3.23415
# ca = 1.5992

# vasp_create_hcp_basal(a, ca, np.array([1, 1, 10]) )
# vasp_create_hcp_prism1(a, ca, np.array([1, 1, 16]) )
# vasp_create_hcp_pyr1(a, ca, np.array([1, 1, 16]), bp=-33)
# vasp_create_hcp_pyr1(a, ca, np.array([1, 1, 16]), bp=33)
# vasp_create_hcp_pyr2(a, ca, np.array([1, 1, 10]) )



