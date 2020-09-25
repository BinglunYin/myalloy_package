#!/home/yin/opt/bin/python3


import numpy as np
from myvasp import vasp_func as vf



def shift_to_complete_layers():
    nlayers, nmiss = check_layers()

    if nmiss > 0.1:
        print('==> shift to POSCAR_layer, nmiss:', nmiss)
        shift_to_poscar_layer(nmiss)
    else:
        print('==> no shift needed. ')





#==========================

def check_layers(filename='CONTCAR'):
    atoms = vf.my_read_vasp(filename = filename)

    z = atoms.get_positions()[:,-1]
    z.sort()
    natoms = len(z)
    
    dz = np.diff(z)

    if dz.min() > dz.max()*0.9:
        print('==> look like 1 atom per layer.')
        nlayers = natoms
        nmiss = 0

    else:
        if dz.max() > 5:
            print('==> look like having a vacuum layer.')     
            dz_maxid = np.argmax(dz)
            temp = dz.copy()
            temp.sort()
            dz[ dz_maxid ] = temp[-2] + 1e-8
        else:
            print('==> look like normal supercell. ')
        
        dz_a, dz_b = k_means(dz) 
        print('==> dz_a[-5:], dz_b:', dz_a[-5:], dz_b) 

        natomsl, nlayers, nmiss = \
            calc_natomsl_nlayers_nmiss(dz, dz_b)
    
    return nlayers, nmiss



def k_means(dz):
    dz = dz.copy()
  
    mask = ( dz == dz.max() ) 
    a = dz[ np.invert(mask)  ]     # other dz
    b = dz[ mask ]                 # largest dz

    while True:  
        mask = ( np.abs(a-a.mean()) <=  np.abs(a-b.mean()) )
        a_t = a[ mask ]
        a_f = a[ np.invert(mask) ]
    
        mask = ( np.abs(b-b.mean()) <=  np.abs(b-a.mean()) )
        b_t = b[ mask ]
        b_f = b[ np.invert(mask) ]
        print(a_t.shape, a_f.shape, b_t.shape, b_f.shape )

        if len(a_f)==0 and len(b_f)==0 :  
            break  

        a = np.hstack([ a_t, b_f ])
        b = np.hstack([ b_t, a_f ])

    a.sort()
    b.sort()
    return a, b



def calc_natomsl_nlayers_nmiss(dz, dz_b):
    dz = dz.copy()
    dz_b = dz_b.copy()

    # interplane id in dz
    intp_id = np.array([])
    for item in np.unique(dz_b):
        temp, = np.where(dz == item)
        intp_id = np.append(intp_id, temp)
    
    intp_id.sort()

    d_intp_id = np.diff(intp_id)
    vf.confirm_int(d_intp_id)

    # use the most frequent value as the natoms per layer
    natomsl = np.bincount( d_intp_id.astype(int) ).argmax() 
    # natomsl = calc_natomsl(d_intp_id)


    vf.confirm_int( d_intp_id/natomsl  )

    natoms = len(dz)+1
    nlayers = natoms / natomsl 

    temp = np.ceil( (intp_id[0]+1)/natomsl )
    nmiss = natomsl*temp - (intp_id[0]+1) 
    
    vf.confirm_int( [natomsl, nlayers, nmiss] )
    print('==> natomsl, nlayers, nmiss:', \
        natomsl, nlayers, nmiss)
    return natomsl, nlayers, nmiss



def calc_natomsl(d_intp_id):
    print(d_intp_id)

    d_intp_id_uniq = np.unique(d_intp_id)
    k = np.array([])
    
    for i in d_intp_id_uniq:
        mask = (d_intp_id == i)
        temp = len( d_intp_id[mask] )
        k = np.append(k, temp)

    natomsl = d_intp_id_uniq[ np.argmax(k) ]
    return natomsl







#==========================

def shift_to_poscar_layer(nmiss):
    atoms = vf.my_read_vasp(filename='CONTCAR')

    z = atoms.get_positions()[:,-1]
    z.sort()
    natoms = len(z)

    latt33 = atoms.get_cell()[2, 2]

    temp1 = latt33 - z[int(-1*nmiss)]
    temp2 = z[int(-1*nmiss)] - z[int(-1*nmiss-1)] 

    zshift = temp1 + 0.5*temp2

    pos = atoms.get_positions()
    pos[:,2] = pos[:,2] + zshift
    atoms.set_positions(pos)
    atoms.wrap()

    vf.my_write_vasp(atoms, filename='POSCAR_layer', vasp5=True)


    # check
    atoms2 = vf.my_read_vasp('POSCAR_layer')
    atoms3 = vf.my_read_vasp('CONTCAR')
    
    latt = atoms2.get_cell()[:]
    vf.confirm_0( latt-atoms3.get_cell()[:] ) 
    
    dpos  = atoms2.get_positions() - atoms3.get_positions() 
    
    dposD = dpos @ np.linalg.inv(latt)
    dposD = dposD - np.around(dposD)  
    dpos = dposD @ latt

    for i in np.arange(3):
        dpos[:,i] = dpos[:,i] - dpos[:,i].mean()
    
    vf.confirm_0(dpos) 



# shift_to_complete_layers()



