
#!/home/yin/opt/bin/python3

import numpy as np
import sys, copy, os
from myvasp import vasp_func as vf 


# y_post_EPI.X.txt  - X
# y_post_EPI.Ef.txt - Ef




def calc_X_Ef_from_y_dir(shellmax=20):
    # this runs outside y_dir 
    # y_post_EPI.Ef.txt - Ef
    
    latoms = vf.get_list_of_atoms()
    calc_X_from_latoms(latoms, shellmax=shellmax)

    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    njobs = len(jobn)

    Ef = np.array([])  
    for i in np.arange(njobs):
        natoms = latoms[i].get_positions().shape[0]
        Ef = np.append(Ef, Etot[i]/natoms )

    vf.confirm_0( Ef.shape - np.array([njobs, ])) 
    np.savetxt("y_post_EPI.Ef.txt", Ef )






def calc_X_from_poscars2(dirname='poscars2', shellmax=20):
    # this runs outside the dir 
    
    latoms = vf.get_list_of_atoms_from_poscars2(dirname=dirname)
    calc_X_from_latoms(latoms, shellmax=shellmax)






#================================================


def calc_X_from_latoms(latoms_in, shellmax=20):
    # from list of atoms
    # y_post_EPI.X.txt  - X
    
    import pandas as pd 
    
    latoms = copy.deepcopy( latoms_in )
    njobs = len(latoms)

    # check, cn and elem_sym need to be the same, natoms doesn't     
    cn = latoms[0].get_cn()
    nelem = len(cn) 
    elem_an  = pd.unique( latoms[0].get_atomic_numbers()   )
    elem_sym = pd.unique( latoms[0].get_chemical_symbols() )

    dn_shell_row_all = calc_dn_shell_row(latoms[0], shellmax=shellmax, write_dn=False)


    for i in np.arange(1, njobs):
        vf.confirm_0(latoms[i].get_cn() - cn)

        temp = pd.unique( latoms[i].get_atomic_numbers() )
        vf.confirm_0( temp - elem_an )

        temp = pd.unique( latoms[i].get_chemical_symbols() )
        if elem_sym.any() != temp.any() :
            sys.exit('ABORT: wrong elem_sym. ')

        temp = calc_dn_shell_row(latoms[i], shellmax=shellmax, write_dn=False)
        dn_shell_row_all = np.vstack([ dn_shell_row_all, temp ])


    vf.confirm_0( dn_shell_row_all.shape - np.array([njobs, (nelem)*(nelem-1)/2*shellmax ]) )
  
    X = dn_shell_row_all*(-1/2)
    temp = np.ones([ X.shape[0], 1 ])
    X = np.hstack([ temp, X]) 

    np.savetxt("y_post_EPI.X.txt", X )






#================================================


def calc_dn_shell_row(atoms_in, shellmax=20, write_dn=False):
    atoms = copy.deepcopy(atoms_in)

    natoms = atoms.get_positions().shape[0]
    V0 = atoms.get_volume() / natoms
    a0 = (V0*4)**(1/3)
    cn = atoms.get_cn() 
    print('==> shellmax: ', shellmax, '; a0: ', a0, '; cn: ', cn )

    cc_scale = calc_cc_scale(cn)


    vf.my_write_vasp(atoms, filename='CONTCAR_for_ovito', vasp5=True)

    # struc = calc_ovito_cna(atoms)
    rcrys, ncrys = vf.crystal_shell('fcc')

    cutoff = a0 * rcrys[shellmax-1] 
    cutoff =  np.ceil(cutoff*1e3) /1e3

    data_rdf = calc_ovito_rdf(cutoff)
    r, n = post_rdf(data_rdf, V0, cc_scale)

    ncrys_tot = ncrys[0:shellmax].sum()
    while  np.abs( n.sum() - ncrys_tot) >1e-10 :

        if n.sum() > ncrys_tot :
            sys.exit('==> ABORT. impossible cutoff. ')

        cutoff = cutoff + 1e-2 
        data_rdf = calc_ovito_rdf(cutoff)
        r, n = post_rdf(data_rdf, V0, cc_scale)

    os.remove('CONTCAR_for_ovito') 

    # convert n(r) to n(shell) 
    dn_shell, sro_shell = calc_n_shell(ncrys, shellmax, n, cc_scale)


    # remove like-pairs
    rmid = calc_rmid(cn)
    dn_shell_r  = calc_reduced_dn_shell(dn_shell,  rmid)
    sro_shell_r = calc_reduced_dn_shell(sro_shell, rmid)



    # convert to a row 
    dnw = np.prod(dn_shell_r.shape)            # dn width
    dn_shell_row  = dn_shell_r.reshape(dnw) /2
    sro_shell_row = sro_shell_r.reshape(dnw) 

    if write_dn == True:
        np.savetxt("y_post_dn_shell_row.txt", dn_shell_row )       # Delta_eta
        np.savetxt("y_post_sro_shell_row.txt", sro_shell_row )     # WC SRO

    return dn_shell_row 






#================================================


def calc_cc_scale(cn):
    # scaling factors, e.g., c1c1, 2c1c2, c2c2
    cn2 = cn[np.newaxis, :].copy()
    cn2prod = cn2.T @ cn2
      
    nelem = cn.shape[0]
    cc_scale = np.array([])

    for i in np.arange(nelem):
        for j in np.arange(i, nelem):
            temp = cn2prod[i, j]
            if i != j :
                temp = temp*2
            cc_scale = np.append(cc_scale, temp)

    return cc_scale




def calc_ovito_cna():
    # from ovito.pipeline import StaticSource, Pipeline
    # from ovito.io.ase import ase_to_ovito
    from ovito.io import import_file
    from ovito.modifiers import CommonNeighborAnalysisModifier
    
    print('==> running CNA in ovito ')
    pipeline = import_file('CONTCAR_for_ovito')

    modifier = CommonNeighborAnalysisModifier()
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    cna = data.tables['structures'].xy() 
    mask = cna[:,1]>0.5
    cna1 = cna[mask]
    if cna1.shape[0] != 1:
        sys.exit('ABORT: wrong CNA. ')

    ovito_struc = ['other', 'fcc', 'hcp', 'bcc', 'ico']
    struc = ovito_struc[ cna1[0, 0] ]
    return struc




def calc_ovito_rdf(cutoff):
    # from ovito.pipeline import StaticSource, Pipeline
    # from ovito.io.ase import ase_to_ovito
    from ovito.io import import_file
    from ovito.modifiers import CoordinationAnalysisModifier

    print('==> cutoff in ovito rdf: {0}'.format(cutoff))    
    pipeline = import_file('CONTCAR_for_ovito')

    modifier = CoordinationAnalysisModifier(
        cutoff = cutoff, number_of_bins = 200, partial=True)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    data_rdf = data.tables['coordination-rdf'].xy()

    dr = data_rdf[1,0] - data_rdf[0,0]
    vf.confirm_0( dr - cutoff/200, str1='wrong dr')

    return data_rdf

    


def post_rdf(data_rdf, V0, cc_scale):
    data = data_rdf.copy()

    r = data[:,0].copy()
    dr = r[1] - r[0]

    temp = np.diff(r)
    vf.confirm_0( temp-dr, str1='wrong dr 2')  

    n = data[:, 1:].copy()   # number of neighbours for each pair

    temp = 1/V0 * 4/3*np.pi * ((r+dr/2)**3 - (r-dr/2)**3)
    n = (n.T * temp).T

    vf.confirm_0( cc_scale.shape[0] - n.shape[1],  str1='wrong scaling' )
    n = n * cc_scale

    return r, n
  



def calc_n_shell(ncrys, shellmax, n, cc_scale):
    ntot = np.sum(n, axis=1)

    # find the index of critical r separating shells
    rc = np.array([0])
    for i in np.cumsum( ncrys[0:shellmax] ):
        for j in np.arange(1, len(ntot) ):
            if ( np.abs(np.cumsum(ntot)[j-1] -i) < 1e-10 ) \
                and ( np.abs(np.cumsum(ntot)[j] -i) < 1e-10 ):
                rc = np.append(rc, j)
                break

    if np.abs( rc.shape[0] -(shellmax+1) ) > 1e-10 :
        if np.abs( rc.shape[0] - shellmax ) < 1e-10:
            rc = np.append(rc, len(ntot) )
        else:
            sys.exit('==> ABORT. wrong rc. {0}'.format(rc) )

    # number of each pair in each shell
    n_shell=np.zeros([shellmax, n.shape[1]])
    for i in np.arange(shellmax):
        n_shell[i,:] = np.sum( n[rc[i]:rc[i+1], :], axis=0)

    # for ideal random
    n_shell_rand = np.tile( cc_scale, [shellmax, 1] )
    n_shell_rand = ( n_shell_rand.T * ncrys[0:shellmax] ).T

    dn_shell = n_shell - n_shell_rand

    sro_shell = - dn_shell / n_shell_rand

    vf.confirm_0( np.sum(n_shell,  axis=1) - ncrys[0:shellmax], str1='wrong n_shell'  )
    vf.confirm_0( np.sum(dn_shell, axis=1),                     str1='wrong dn_shell' )

    return  dn_shell, sro_shell




def calc_rmid(cn):
    nelem = cn.shape[0]
    rmid = np.array([])    # rm id for like-pairs cn*cn
    k=-1
    for i in np.arange(nelem):
        for j in np.arange(i, nelem):
            k = k+1
            if i == j :
                rmid = np.append(rmid, k)
    return rmid




def calc_reduced_dn_shell(dn_shell_in, rmid):
    dn_shell = dn_shell_in.copy()
    if len(dn_shell.shape) < 1.9:      # for binary 
        dn_shell = dn_shell[np.newaxis, :]
    dn_shell_r = np.delete(dn_shell, rmid.astype(int), axis=1)
    return dn_shell_r





