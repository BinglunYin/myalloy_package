

from myvasp import vasp_func as vf 
import copy, sys
import numpy as np 




def confirm_111_o_bulk_latt(latoms_in, sd):
    latoms = copy.deepcopy(latoms_in)

    njobs = len(latoms)
    a = latoms[0].pos_a0
    b = a/np.sqrt(2)

    latt_sd = np.array([  
        [ sd[0],                0,                0 ], 
        [     0, np.sqrt(3)*sd[1],                0 ], 
        [     0,                0, np.sqrt(6)*sd[2] ], 
    ]) *b 

    for i in np.arange(njobs):
        vf.confirm_0( latoms[i].cell[:] - latt_sd )
         





def routine_1(latoms_1, latoms_2, dlatt_type='zero', relax_type='relaxed'):
    confirm_dlatt(latoms_1, latoms_2, dlatt_type=dlatt_type )
    confirm_same_size_cn( latoms_1, latoms_2 )
    lurms = calc_lurms(latoms_1, latoms_2 ) 

    if relax_type == 'relaxed':
        print('max urms:', lurms.max() ) 
    elif relax_type == 'unrelaxed':
        vf.confirm_0(lurms)

    return lurms 
    
       



#===============================


def confirm_dlatt(latoms1_in, latoms2_in, dlatt_type='zero'):
    latoms1 = copy.deepcopy(latoms1_in)
    latoms2 = copy.deepcopy(latoms2_in)
    njobs = len(latoms1)
    a = latoms1[0].pos_a0
    b = a/np.sqrt(2)

    bp1 = np.array([ b/2, b/2/np.sqrt(3), 0 ])
    bp2 = np.array([ b, 0, 0 ])

    for i in np.arange(njobs):
        latt1 = latoms1[i].cell[:] 
        latt2 = latoms2[i].cell[:] 
        dlatt = latt2 - latt1 
        
        if dlatt_type == 'zero':
            vf.confirm_0( dlatt )

        elif dlatt_type == 'ssf':            
            vf.confirm_0(dlatt[0:2,:])
            vf.confirm_0(dlatt[2,:] - bp1)

        elif dlatt_type == 'tilt':            
            vf.confirm_0(dlatt[0:2,:])
            vf.confirm_0(dlatt[2,:] - bp2)

        else:
            sys.exit('ABORT: wrong dlatt_type. ')






def confirm_same_size_cn(latoms1_in, latoms2_in):
    latoms1 = copy.deepcopy(latoms1_in)
    latoms2 = copy.deepcopy(latoms2_in)

    njobs = len(latoms1)
  
    for i in np.arange(njobs):
        # check atomic number
        vf.confirm_0( latoms1[i].get_atomic_numbers() \
                    - latoms2[i].get_atomic_numbers() )

        # check formula
        if     latoms1[i].get_chemical_formula() \
            != latoms2[i].get_chemical_formula():
            sys.exit('ABORT: wrong chemical formula. ')

        #check cn
        vf.confirm_0( latoms1[i].get_cn() - latoms2[i].get_cn() )



def confirm_unrelaxed(latoms1_in):
    latoms1 = copy.deepcopy(latoms1_in)
    njobs = len(latoms1)

    latt_ref = latoms1[0].cell[:]
    pos_ref  = latoms1[0].get_positions()

    posD_ref =  sort_pos( calc_posD(pos_ref, latt_ref) )

    for i in np.arange(njobs):  
        latt = latoms1[i].cell[:]
        vf.confirm_0(latt - latt_ref)
       
        pos = latoms1[i].get_positions()
        posD = sort_pos( calc_posD(pos, latt) )
        vf.confirm_0(posD - posD_ref)



def calc_posD(pos, latt):
    posD = pos @ np.linalg.inv(latt)
    posD = posD - np.around(posD) 
    return posD 



def sort_pos(pos_in):
    pos = pos_in.copy()
    natoms = pos.shape[0]

    for j in np.arange(3):
        for i in np.arange(natoms-1, 0, -1): 
            for k in np.arange(i):
                if pos[k,j] > pos[k+1,j]:         # move largest to the end
                    temp = pos[k,:].copy() 
                    pos[k,:] = pos[k+1,:].copy() 
                    pos[k+1,:] = temp.copy() 
    return pos 


    


# lattice distortion

def calc_lurms(latoms1_in, latoms2_in):
    latoms1 = copy.deepcopy(latoms1_in)
    latoms2 = copy.deepcopy(latoms2_in)
    njobs = len(latoms1)
    
    lurms = np.array([])

    for i in np.arange(njobs):
        u, urms = calc_urms( latoms1[i], latoms2[i])
        lurms = np.append(lurms, urms)
    
        if u.max() > 0.5:
            print('\nWARNING: u.max() is larger than 0.5 Ang!')
            print('i, u.max(), urms :')
            print( i, u.max(), urms)

    return lurms




    
def calc_urms(atoms1_in, atoms2_in):
    atoms1 = copy.deepcopy(atoms1_in)
    atoms2 = copy.deepcopy(atoms2_in)
        
    latt1 = atoms1.cell[:]
    latt2 = atoms2.cell[:]
    # vf.confirm_0( latt1 - latt2 ) 

    pos1  = atoms1.get_positions()
    pos2  = atoms2.get_positions()

    dpos = pos2 - pos1

    dposD = calc_posD(dpos, latt2)    
    dpos = dposD @ latt2  

    rigid_shift = np.mean(dpos, axis=0)
    if rigid_shift.shape != (3,):
        sys.exit('ABORT: wrong rigid_shift.')
    dpos = dpos - rigid_shift
    vf.confirm_0( dpos.sum() )
 
    u = np.linalg.norm( dpos, axis=1) 

    if u.shape != (dpos.shape[0],):
        sys.exit('ABORT: wrong u.')

    urms = np.sqrt( np.mean(u**2) )

    return u, urms
    
    
    


