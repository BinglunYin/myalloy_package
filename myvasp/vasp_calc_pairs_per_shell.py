#!/home/yin/opt/bin/python3

import numpy as np
import sys, os, copy
from myvasp import vasp_func as vf 



def calc_pairs_per_shell_from_CONTCAR(shellmax = 4):

    atoms = vf.my_read_vasp('CONTCAR')
    calc_pairs_per_shell(atoms, shellmax = shellmax)





def calc_pairs_per_shell(atoms_in, shellmax=4, write_dp=True):
    atoms = copy.deepcopy(atoms_in)

    print('==> shellmax: ', shellmax)
    vf.my_rm('y_post_dp_shell.txt')

    natoms = atoms.get_positions().shape[0]
    V0 = atoms.get_volume() / natoms
    a0 = (V0*4)**(1/3)
    cn = atoms.cn
    print('==> a0: ', a0, '; cn: ', cn )

    cc_scale = calc_cc_scale(cn)


    vf.my_write_vasp(atoms, filename='CONTCAR_for_ovito', vasp5=True)

    struc = calc_ovito_cna(atoms)
    if struc == 'fcc':
        print('==> fcc ')
        rcrys, ncrys = fcc_shell()
    elif struc == 'hcp':
        print('==> hcp ')
        rcrys, ncrys = hcp_shell()


    cutoff = np.around( a0 * rcrys[shellmax-1], 1)
    data_rdf = calc_ovito_rdf(atoms, cutoff)
    r, n = post_rdf(data_rdf, V0, cc_scale)

    while np.abs( n.sum() - ncrys[0:shellmax].sum() ) > 1e-10:
        if n.sum() > ncrys[0:shellmax].sum() :
            sys.exit('==> ABORT. impossible cutoff. ')
        cutoff = cutoff+0.1
        data_rdf = calc_ovito_rdf(atoms, cutoff)
        r, n = post_rdf(data_rdf, V0, cc_scale)

    dn_shell, WC_SRO = calc_n_shell(ncrys, shellmax, r, n, cc_scale)


    rmid = calc_rmid(cn)
    dn_shell_r = read_reduced_dn_shell(dn_shell, rmid)
    WC_SRO_r = read_reduced_dn_shell(WC_SRO, rmid)

    dpw = np.prod(dn_shell_r.shape)  # dp width
    dp_shell  = dn_shell_r.reshape(dpw) /2
    WC_SRO_r2 =   WC_SRO_r.reshape(dpw) 

    if write_dp == True:
        np.savetxt("y_post_dp_shell.txt", dp_shell )
        np.savetxt("y_post_WC_SRO_shell.txt", WC_SRO_r2 )

    return dp_shell







def fcc_shell():
    rfcc = np.sqrt(  np.array([ 
        1/2, 1,   3/2, 2,   5/2, 3,    
        7/2, 4,   9/2, 5,   11/2, 6,   13/2    ])  ) # in unit of a0
    nfcc = np.array([ 
        12, 6,   24, 12,   24, 8,
        48, 6,   36, 24,   24, 24,    72   ])
    return rfcc, nfcc




def hcp_shell():
    rhcp = np.array([
        0.707, 0.999, 1.154, 1.224, 1.354, \
        1.414, 1.581, 1.683, 1.732, 1.779, \
        1.825, 1.870, 1.914,  ])
    nhcp = np.array([
        12,  6,  2, 18, 12,  \
         6, 12, 12,  6,  6,  \
        12, 24,  6,  ])
    return rhcp, nhcp




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




def calc_ovito_cna(atoms_in):
    # from ovito.pipeline import StaticSource, Pipeline
    # from ovito.io.ase import ase_to_ovito
    from ovito.io import import_file
    from ovito.modifiers import CommonNeighborAnalysisModifier
    
    print('==> running CNA in ovito ')

    # atoms = copy.deepcopy(atoms_in)
    # data = ase_to_ovito(atoms)
    # pipeline = Pipeline(source = StaticSource(data = data))
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




def calc_ovito_rdf(atoms_in, cutoff = 6.0):
    # from ovito.pipeline import StaticSource, Pipeline
    # from ovito.io.ase import ase_to_ovito
    from ovito.io import import_file
    from ovito.modifiers import CoordinationAnalysisModifier

    print('==> cutoff in ovito rdf: {0}'.format(cutoff))
    
    # atoms = copy.deepcopy(atoms_in)
    # data = ase_to_ovito(atoms)
    # pipeline = Pipeline(source = StaticSource(data = data))
    pipeline = import_file('CONTCAR_for_ovito')

    modifier = CoordinationAnalysisModifier(
        cutoff = cutoff, number_of_bins = 200, partial=True)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    # np.savetxt("y_post_ovito_rdf.txt", 
        # data.tables['coordination-rdf'].xy() )
    data_rdf = data.tables['coordination-rdf'].xy()
    return data_rdf

    

def post_rdf(data_rdf, V0, cc_scale):
    data = data_rdf.copy()

    r = data[:,0].copy()
    dr = r[1] - r[0]
    temp = r[-1] - r[-2]
    if np.abs(dr - temp)>1e-10:
        sys.exit('==> ABORT. wrong dr. {0}'.format([dr, temp]))

    n = data[:, 1:].copy()   # number of neighbours for each pair

    temp = 1/V0 * 4/3*np.pi * ((r+dr/2)**3 - (r-dr/2)**3)
    n = (n.T * temp).T

    if np.abs(cc_scale.shape[0] - n.shape[1]) > 1e-10 :
        sys.exit('==> ABORT. wrong scaling. ')
    n = n * cc_scale

    return r, n
  



def calc_n_shell(ncrys, shellmax, r, n, cc_scale):
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

    WC_SRO = - dn_shell / n_shell_rand

    if np.linalg.norm( np.sum(n_shell, axis=1) - ncrys[0:shellmax] )  > 1e-10:
        sys.exit('==> wrong n_shell')
      
    if np.linalg.norm( np.sum(dn_shell, axis=1)  )  > 1e-10:
        sys.exit('==> wrong dn_shell')

    # np.savetxt("y_post_n_shell.txt",   n_shell )
    # np.savetxt("y_post_dn_shell.txt", dn_shell )
    return  dn_shell, WC_SRO





def calc_rmid(cn):
    nelem = cn.shape[0]
    rmid = np.array([])    # rm id for cn*cn
    k=-1
    for i in np.arange(nelem):
        for j in np.arange(i, nelem):
            k = k+1
            if i == j :
                rmid = np.append(rmid, k)
    # print('rmid:', rmid)
    return rmid





def read_reduced_dn_shell(dn_shell_in, rmid):
    dn_shell = dn_shell_in.copy()
    if len(dn_shell.shape) < 1.9:
        dn_shell = dn_shell[np.newaxis, :]
    dn_shell_r = np.delete(dn_shell, rmid.astype(int), axis=1)
    return dn_shell_r







# calc_pairs_per_shell_from_CONTCAR()



