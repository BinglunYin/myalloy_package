

import numpy as np 
import os, sys, copy 
from myvasp import vasp_func as vf 





class class_lpairs:
    def __init__(self, r_shell, leta):
        self.r_shell = r_shell 
        self.leta = leta 
        self.auto_add()


    def auto_add(self):        
        self.njobs = len(self.leta) 

        temp = np.array(self.leta[0].shape)
        self.shellmax = temp[0]      # all the shells computed in ovito 
        self.nelem    = temp[1]
        vf.confirm_0(   temp[2] - temp[1] )

        for i in np.arange(self.njobs):
            vf.confirm_0( np.array(self.leta[i].shape) - temp )         

        self.calc_leta2() 
        self.calc_lepsi() 
        print('==> lparis created!\n==> njobs, shellmax, nelem:', \
            self.njobs, self.shellmax, self.nelem )


    def calc_leta2(self):
        # reshape leta to 2D, without eta_nn, for get_x  
        leta2 = []
        for k in np.arange( self.njobs ):
            eta2 = np.zeros([ self.shellmax, int(self.nelem*(self.nelem-1)/2) ])
            for d in np.arange(self.shellmax):
                m = -1
                for i in np.arange(self.nelem -1):
                    for j in np.arange(i+1, self.nelem):
                        m = m+1 
                        eta2[d,m] = self.leta[k][d, i, j]
            leta2.append(eta2)
        self.leta2 = leta2 
            

    def calc_lepsi(self): 
        lepsi = [] 
        for k in np.arange(self.njobs):
            epsi = np.sum(self.leta[k], axis=2)
            lepsi.append(epsi) 
        self.lepsi = lepsi


    #-----------------------------------

    def get_X(self, dmax):           
        X = self.get_x(self.leta2[0], self.lepsi[0], dmax)         
        for i in np.arange(1, self.njobs ):        
            x = self.get_x(self.leta2[i], self.lepsi[i], dmax)        
            X = np.vstack([ X, x ])                
        return X 


    def get_x(self, eta2, epsi, dmax):   
        # for each eta, epsi
        eta2_x = eta2[0:dmax]
        epsi_x = epsi[0:dmax, 1:]    

        eta2_x = eta2_x.reshape( int(np.prod(eta2_x.shape)) )
        epsi_x = epsi_x.reshape( int(np.prod(epsi_x.shape)) )

        temp1 = np.append( eta2_x*(-0.5), 1 )
        temp2 = np.append( temp1, epsi_x*(0.5) )  
        return temp2  
      







#=================================
# calc list of pairs   
#=================================

def calc_lpairs_from_poscars2(dirname='poscars2'):
    # from dir, runs outside the dir 
    latoms = vf.get_list_of_atoms_from_poscars2(dirname=dirname)
    calc_lpairs_from_latoms(latoms)




def calc_lpairs_from_latoms(latoms_in):
    # from list of atoms   
    import pandas as pd 

    latoms = copy.deepcopy( latoms_in )
    njobs = len(latoms)

    # check, elem_sym need to be the same. cn, natoms, doesn't matter      
    elem_an  = pd.unique( latoms[0].get_atomic_numbers()   )
    elem_sym = pd.unique( latoms[0].get_chemical_symbols() )

    lr_shell = [] 
    leta = [] 
    for i in np.arange(njobs):
        temp = pd.unique( latoms[i].get_atomic_numbers() )
        vf.confirm_0( temp - elem_an )

        temp = pd.unique( latoms[i].get_chemical_symbols() )
        if (elem_sym != temp).any() :
            sys.exit('ABORT: wrong elem_sym. ')

        temp_r, temp = calc_eta(latoms[i])
        lr_shell.append(temp_r)  
        leta.append(temp) 

    for i in np.arange(njobs):
        vf.confirm_0( lr_shell[i] - lr_shell[0] ) 

    lpairs1 = class_lpairs( lr_shell[0], leta )
    vf.my_save_pkl(lpairs1, 'lpairs.pkl')








#=================================
# calc eta for one realization  
#=================================


def calc_eta(atoms_in):
    atoms = copy.deepcopy( atoms_in )
    
    natoms = atoms.get_positions().shape[0]
    V0 = atoms.get_volume() / natoms
    a0 = (V0*4)**(1/3)
    cn = atoms.get_cn() 
    nelem = atoms.get_nelem() 
   
    cc_scale = calc_cc_scale(cn)

    r_fcc, n_fcc = vf.crystal_shell('fcc')   

    cutoff = np.ceil(r_fcc[19]*a0) 
    vf.my_write_vasp(atoms, filename='CONTCAR_for_ovito', vasp5=True)
    data_rdf = calc_ovito_rdf(cutoff=cutoff)
    r, n = calc_n_from_rdf(data_rdf, V0, cc_scale)    
    os.remove('CONTCAR_for_ovito') 
    
    # convert n(r) to n(shell) 
    r_shell, n_shell = calc_n_shell(r, n)
    
    # convert to eta, 3d array, d*n*n
    n3 = calc_n3(n_shell, nelem)
        
    vf.confirm_0( np.sum(n_shell, axis=1)[0:2]            - n_fcc[0:2], str1='wrong n_shell') 
    vf.confirm_0( np.sum(np.sum(n3, axis=2), axis=1)[0:2] - n_fcc[0:2], str1='wrong n3')       
    return r_shell, n3 




#---------------------------------

def calc_cc_scale(cn):
    # scaling factors, e.g., c1c1, 2c1c2, c2c2
    nelem = cn.shape[0]
    cn2 = cn[np.newaxis, :].copy()
    cn2prod = cn2.T @ cn2    

    cc_scale = np.array([])
    for i in np.arange(nelem):
        for j in np.arange(i, nelem):
            temp = cn2prod[i, j]
            if i != j :
                temp = temp*2
            cc_scale = np.append(cc_scale, temp)
    return cc_scale




def calc_ovito_rdf(cutoff):
    from ovito.io import import_file
    from ovito.modifiers import CoordinationAnalysisModifier

    print('==> cutoff in ovito rdf: {0}'.format(cutoff))    
    pipeline = import_file('CONTCAR_for_ovito')

    nbin = int(cutoff*1000)
    modifier = CoordinationAnalysisModifier(
        cutoff=cutoff, number_of_bins=nbin, partial=True)
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()

    data_rdf = data.tables['coordination-rdf'].xy() 
    vf.confirm_0( np.diff(data_rdf[:,0]) - cutoff/nbin, str1='wrong dr')
    return data_rdf




def calc_n_from_rdf(data_rdf, V0, cc_scale):
    data = data_rdf.copy()

    r = data[:,0].copy()
    dr = r[1] - r[0]
    vf.confirm_0( np.diff(r) - dr, str1='wrong dr')

    n = data[:, 1:].copy()   # number of neighbours for each pair, as a func of r 
    temp = 1/V0 * 4/3*np.pi * ( (r+dr/2)**3 - (r-dr/2)**3 )
    n = (n.T * temp).T

    vf.confirm_0( cc_scale.shape[0] - n.shape[1],  str1='wrong cc_scale' )
    n = n * cc_scale 
    return r, n
      



def calc_n_shell(r, n):
    ntot = np.sum(n, axis=1) 

    r_shell = np.array([])
    n_shell = np.zeros([1, n.shape[1]])
    k       = np.array([]) 
    for i in np.arange(1, len(r)):
        if ntot[i] != 0:
            if ntot[i-1] == 0:   # new peak
                r_shell = np.append(r_shell, r[i])
                n_shell = np.vstack([n_shell, n[i]])
                k       = np.append(k, 1)
            else:                # not new, part of last peak   
                r_shell[-1] = r_shell[-1] + r[i]
                n_shell[-1] = n_shell[-1] + n[i]
                k[-1]       = k[-1]       + 1

    r_shell = r_shell / k 
    n_shell = np.delete(n_shell, 0, 0)
    vf.confirm_0( r_shell.shape[0] - n_shell.shape[0], str1='wrong r_shell')
    return r_shell, n_shell




def calc_n3(n_shell, nelem):
    shellmax = n_shell.shape[0]     
    vf.confirm_0( n_shell.shape[1] - (nelem+1)*nelem/2, str1='wrong nelem ' )

    n3 = np.zeros([shellmax, nelem, nelem])
    for d in np.arange(shellmax):
        m = -1
        for i in np.arange(nelem):
            for j in np.arange(i, nelem):
                m = m+1  
                if i==j:
                    k = 1
                else:
                    k = 2 
                n3[d, i, j] = n_shell[d, m] /k 
                n3[d, j, i] = n_shell[d, m] /k     
    return n3 

#=================================


     



