

import numpy as np
import sys, copy, os 
from myvasp import vasp_func as vf
from myvasp import vasp_EPI_dp_shell as vf2
import pandas as pd



# =====================================
# compare structures between two sets
# =====================================


def check_elem(latoms1_in, latoms2_in):
    latoms1 = copy.deepcopy(latoms1_in)
    latoms2 = copy.deepcopy(latoms2_in)
    njobs = len(latoms1)

    cn = latoms1[0].get_cn()
  
    for i in np.arange(njobs):
        #check cn
        vf.confirm_0( latoms1[i].get_cn() - cn )
        vf.confirm_0( latoms2[i].get_cn() - cn )

        # check elem
        if latoms1[i].get_chemical_formula() \
            != latoms2[i].get_chemical_formula():
            sys.exit('ABORT: wrong chemical formula. ')

        # check atomic number
        temp = latoms1[i].get_atomic_numbers() \
            - latoms2[i].get_atomic_numbers()
        vf.confirm_0( temp )






def check_latt(latoms1_in, latoms2_in, latt_type='same', k=1/4):
    latoms1 = copy.deepcopy(latoms1_in)
    latoms2 = copy.deepcopy(latoms2_in)
    njobs = len(latoms1)

    for i in np.arange(njobs):
        latt1 = latoms1[i].cell[:] 
        latt2 = latoms2[i].cell[:] 
        lattd = latt2 - latt1 
        
        if latt_type == 'same':
            vf.confirm_0( lattd )

        elif latt_type == 'tilt':            
            vf.confirm_0(lattd[0:2,:])
            vf.confirm_0(lattd[2,:] - k*latt1[0,:] )

        else:
            sys.exit('ABORT: wrong latt. ')






def check_unrelaxed(latoms1_in):
    latoms1 = copy.deepcopy(latoms1_in)
    njobs = len(latoms1)

    latt_ref = latoms1[0].cell[:]
    pos_ref = latoms1[0].get_positions()

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

def calc_lurms(jobn1, latoms1_in, latoms2_in):
    latoms1 = copy.deepcopy(latoms1_in)
    latoms2 = copy.deepcopy(latoms2_in)
    njobs = len(latoms1)
    
    lurms = np.array([])

    for i in np.arange(njobs):
        u, urms = calc_urms( latoms1[i], latoms2[i])
        lurms = np.append(lurms, urms)
    
        if u.max() > 0.5:
            print('\nWARNING: u.max() is larger than 0.5 Ang!')
            print('jobn1[i], u.max(), urms :')
            print( jobn1[i], u.max(), urms)

    return lurms




    
def calc_urms(atoms1_in, atoms2_in):
    atoms1 = copy.deepcopy(atoms1_in)
    atoms2 = copy.deepcopy(atoms2_in)
        
    latt1 = atoms1.cell[:]
    latt2 = atoms2.cell[:]

    pos1  = atoms1.get_positions()
    pos2  = atoms2.get_positions()

    dpos = pos2 - pos1
    natoms = dpos.shape[0]

    dposD = calc_posD(dpos, latt2)    
    dpos = dposD @ latt2  

    rigid_shift = np.mean(dpos, axis=0)
    if rigid_shift.shape != (3,):
        sys.exit('ABORT: wrong rigid_shift.')
    dpos = dpos - rigid_shift
    vf.confirm_0( dpos.sum() )
 
    u = np.linalg.norm( dpos, axis=1) 
    if u.shape != (natoms,):
        sys.exit('ABORT: wrong u.')

    urms = np.sqrt( np.mean(u**2) )

    return u, urms
    
    
    









# ===========================
# calculate EPIs 
# ===========================


def calc_X_Ef(shellmax):
    # this runs inside the job dir
    # generate y_post_EPI.data0.txt - [X, Ef]

    mycmd = vf2.calc_pairs_per_shell_from_CONTCAR
    vf.run_cmd_in_jobn(mycmd, shellmax=shellmax)

    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    njobs = len(jobn)

    latoms = vf.get_list_of_atoms()

    # cn and elem_sym need to be the same, natoms doesn't 

    cn = latoms[0].get_cn()
    print('cn:', cn)
    nelem = len(cn) 

    elem_sym = pd.unique( latoms[0].get_chemical_symbols() )
    print('elem_sym:', elem_sym)
    
    Ef = np.array([])  
    for i in np.arange(njobs):
        vf.confirm_0(latoms[i].get_cn() - cn)

        temp = pd.unique( latoms[i].get_chemical_symbols() )
        if elem_sym.any() != temp.any() :
            sys.exit('ABORT: wrong elem_sym. ')

        temp = latoms[i].get_positions().shape[0]
        Ef = np.append(Ef, Etot[i]/temp )
       

    dp_shell_tot = np.loadtxt('./y_dir/%s/y_post_dp_shell.txt' %(jobn[0]) )
    for i in np.arange( 1, njobs ):
        filename = './y_dir/%s/y_post_dp_shell.txt' %(jobn[i]) 
        temp = np.loadtxt( filename )
        dp_shell_tot = np.vstack([ dp_shell_tot, temp ]) 

    vf.confirm_0( dp_shell_tot.shape - np.array([njobs, (nelem)*(nelem-1)/2*shellmax ]) )
    vf.confirm_0( Ef.shape - np.array([njobs, ])) 
  
    X = dp_shell_tot*(-1/2)
    temp = np.ones([ X.shape[0], 1 ])
    X = np.hstack([ temp, X]) 

    temp = np.hstack([ X, Ef[:, np.newaxis] ])
    np.savetxt("y_post_EPI.data_raw.txt", temp )






def calc_EPI(EPI_type, shellmax, data_in, ntest, filename=''):
    
    if EPI_type == 'normal':
        calc_normal_EPI(shellmax, data_in, ntest)

    elif EPI_type == 'all':
        calc_all_EPI(shellmax, data_in, ntest)

    elif EPI_type == 'diff':
        calc_diff_EPI(shellmax, data_in, ntest)


    if filename != '':
        n1 = 'y_post_EPI.%s_data.txt' %(filename) 
        n2 = 'y_post_EPI.%s_beta.txt' %(filename) 
        n3 = 'y_post_EPI.%s.pdf'      %(filename) 
    
        os.rename('y_post_EPI.data.txt', n1 )
        os.rename('y_post_EPI.beta.txt', n2 )
        os.rename('y_post_EPI.pdf',      n3 )


    
def calc_normal_EPI(shellmax, data_in, ntest):
    data = copy.deepcopy(data_in)
    X = data[:, 0:-1]
    E = data[:,   -1]

    calc_EPI_kernel(shellmax, X, E, ntest) 



def calc_all_EPI(shellmax, data_in, ntest):
    data = copy.deepcopy(data_in)    
    Xa = np.vstack([ data[0][:, 0:-1], data[1][:, 0:-1] ])
    Ea = np.hstack([ data[0][:,   -1], data[1][:,   -1] ])

    calc_EPI_kernel(shellmax, Xa, Ea, ntest) 



def calc_diff_EPI(shellmax, data_in, ntest):
    data = copy.deepcopy(data_in)   

    X1 = data[0][:, 1:-1]   # skip 1st column 
    E1 = data[0][:, -1]

    X2 = data[1][:, 1:-1] 
    E2 = data[1][:, -1]

    Xd = X2 - X1
    Ed = E2 - E1 

    calc_EPI_kernel(shellmax, Xd, Ed, ntest) 






def calc_EPI_kernel(shellmax, X, Ef, ntest):

    X_train,  X_test  = split_train_test(X,  ntest) 
    Ef_train, Ef_test = split_train_test(Ef, ntest) 

    beta, R2 = vf.mylinreg(X_train, Ef_train)
    np.savetxt("y_post_EPI.beta.txt", beta )

    temp = np.hstack([ X, Ef[:, np.newaxis] ])
    np.savetxt("y_post_EPI.data.txt", temp )
    

    lbeta = len(beta)
    if np.mod( lbeta-1, shellmax) != 0:
        lbeta = lbeta +1
    
    if np.abs( lbeta-1-shellmax ) < 1e-10:    # binary
        plot_EPI(shellmax, X, Ef, ntest, beta)   






def split_train_test(x, ntest):
    if ntest < 0.9:
        x1 = x.copy()
        x2 = []        
    else:        
        temp = int(-1*ntest)
        x1 = x[0: temp].copy()
        x2 = x[temp: ].copy()
        if len( x.shape ) == 1:
            vf.confirm_0( np.hstack([x1, x2]) - x )
        else:
            vf.confirm_0( np.vstack([x1, x2]) - x )
    
    return x1, x2






def plot_EPI(shellmax, X, Ef, ntest, beta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_wh = [7, 2.2]
    fig_subp = [1, 3]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp, fig_sharex=False)

    fig_pos  = np.array([0.1, 0.22, 0.23, 0])
    fig_pos[-1] =  fig_pos[-2]*fig_wh[0]/fig_wh[1]

    for j in np.arange(fig_subp[1]):
        ax1[j].set_position( fig_pos + np.array([0.33*j, 0, 0,  0]) )

    rfcc, nfcc = vf2.crystal_shell('fcc')
    xi = rfcc[0:shellmax].copy()

    # modify EPIs from diff 
    if np.mod(len(beta)-1, shellmax) != 0:
        beta = np.append( 0, beta) 
        temp = np.ones([ X.shape[0], 1 ])
        X = np.hstack([ temp, X]) 

        ax1[1].set_xlabel('$ \\Delta ( E_{f,\\mathrm{DFT} } ) $ (meV/atom)')
        ax1[1].set_ylabel('$ \\Delta ( E_{f,\\mathrm{EPI} } ) $ (meV/atom)')

        ax1[2].set_ylabel('$ \\Delta ( \\Delta \\eta_{\\mathrm{AuNi}, d} ) $')

    else:
        ax1[1].set_xlabel('$ E_{f,\\mathrm{DFT} } - E_{f}^\\mathrm{rand}$ (meV/atom)')
        ax1[1].set_ylabel('$ E_{f,\\mathrm{EPI} } - E_{f}^\\mathrm{rand}$ (meV/atom)')
 
        ax1[2].set_ylabel('$ \\Delta \\eta_{\\mathrm{AuNi}, d}$')


    #=================================

    ax1[0].plot(xi, beta[1:], '-o')
    
    ax1[0].set_xlabel('Pair distance $d/a$')
    ax1[0].set_ylabel('EPI $V_{nm,d}$ (eV)')

    Erand = beta[0]

    if Erand != 0:
        xlim = ax1[0].get_xlim() 
        ylim = ax1[0].get_ylim() 

        str0 = '$E_f^\\mathrm{rand} = %.3f$ eV/atom' %( Erand )  

        ax1[0].text( xlim[0]+ np.diff(xlim)*0.95, ylim[0]+ np.diff(ylim)*0.9, \
            str0, fontsize=5.5, horizontalalignment='right' ) 


    #=================================

    Ef2 = ( Ef     -beta[0] )*1e3
    Ep  = ( X@beta -beta[0] )*1e3   # EPI predicted energy 
    
    Ef2_train, Ef2_test   = split_train_test(Ef2,  ntest) 
    Ep_train,  Ep_test    = split_train_test(Ep,   ntest) 
    
    str1 = 'train, RMSE=%.3f meV/atom' %( vf.calc_RMSE( Ef2_train, Ep_train ) )
    ax1[1].plot( Ef2_train, Ep_train, '.', label=str1 ) 

    if ntest > 0.9:
        str2 = 'test, RMSE=%.3f meV/atom' %( vf.calc_RMSE( Ef2_test, Ep_test ) )
        ax1[1].plot( Ef2_test, Ep_test, '.', label=str2 )

    ax1[1].legend(fontsize=5.5, loc='upper left') 

    xlim = ax1[1].get_xlim() 
    ylim = ax1[1].get_ylim() 
    zlim = np.array([ np.min([ xlim[0], ylim[0] ]),  np.max([ xlim[1], ylim[1] ]) ])

    ax1[1].plot( zlim, zlim, '-k', alpha=0.3)
    ax1[1].set_xlim(zlim) 
    ax1[1].set_ylim(zlim) 

    str3 = '# of structures:\ntrain=%d, test=%d\n\ntrain=$%.3f \pm %.3f$' \
        %( len(Ef2_train), len(Ef2_test), np.mean(Ef2_train), np.std(Ef2_train) )  

    ax1[1].text( zlim[0]+ np.diff(zlim)*0.95, zlim[0]+ np.diff(zlim)*0.1, \
        str3, fontsize=5.5, horizontalalignment='right' ) 


    # fitting error
    e_train = Ef2_train - Ep_train
    temp = np.cov( np.vstack([Ep_train, e_train]),  bias=True)

    vf.confirm_0(temp[0,0]    - np.std(Ep_train)**2  )
    vf.confirm_0(temp[1,1]    - np.std(e_train)**2   )
    vf.confirm_0(np.sum(temp) - np.std(Ef2_train)**2 )

    str3 = 'Ep_train=$%.3f \pm %.3f$\n\ne_train=$%.3f \pm %.3f$' \
        %( np.mean(Ep_train), np.std(Ep_train),  \
           np.mean(e_train),  np.std(e_train) ,  \
        )  




    ax1[1].text( zlim[0]+ np.diff(zlim)*0.05, zlim[0]+ np.diff(zlim)*0.65, \
        str3, fontsize=5.5, horizontalalignment='left' ) 


    #=================================

    for i in np.arange(X.shape[0]):
        ax1[2].plot(xi, X[i, 1:]/(-0.5), '-', alpha=0.3)
    
    x_mean = np.mean(X[:,1:]/(-0.5), axis=0)
    x_std  = np.std( X[:,1:]/(-0.5), axis=0)

    ax1[2].errorbar(xi, x_mean, fmt="ko-", yerr=x_std)
    ax1[2].set_xlabel('Pair distance $d/a$')


    plt.savefig('y_post_EPI.pdf')
    plt.close('all')





