
import numpy as np 
import copy, os

from myvasp import vasp_func as vf 
from myvasp import vasp_EPI_MC as vmc 
from myvasp import vasp_shift_to_complete_layers as vf_shift

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





def plot_gamma_slip(a_fcc=3.840):
    
    ms  = np.array([0, 0])      # mean and std 

    # from 1b to 7b              # debug
    for i in np.arange(7):       
        m1, s1 = calc_gamma_all(a_fcc=a_fcc, b_slip=i+1)
        temp = np.array([ m1, s1 ])
        ms = np.vstack([ms, temp])
        
    filename = 'SRO_gamma_slip.txt'
    np.savetxt(filename, ms)
   

    # plot ms

    fig_wh = [3.15, 2.8]
    fig_subp = [1, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)

    ax1.set_position([0.2, 0.15, 0.75, 0.8])

    xi = np.arange( ms.shape[0] )
    ax1.errorbar(xi, ms[:,0], yerr=ms[:,1], \
        fmt='-o', capsize=1) 

    ax1.set_xlabel('Slip (b)')
    ax1.set_ylabel('Anti-phase boundary $\\gamma_\\mathrm{APB}$ (mJ/m$^2$)')
    
    ax1.set_xticks(xi)

    filename= 'SRO_gamma_slip.pdf' 
    plt.savefig(filename)
    plt.close('all')






# for all structures, sum of npos structures

def calc_gamma_all(a_fcc=3.840, b_slip=1):
    
    EPI_filename='../y_post_EPI.beta_4.txt'
    EPI_beta =  np.loadtxt( EPI_filename )


    os.system('ls POSCAR_step_* > tmp_filelist')   # debug
    f = open('tmp_filelist', 'r')

    gamma_all = np.zeros(2)
    npos = 0

    for line in f:
        npos += 1
        pos_filename = line.strip('\n') 
    
        # for each structure
        gamma_s, nz = calc_gamma_s(EPI_beta, pos_filename, a_fcc, b_slip)
        gamma_all = np.vstack([gamma_all, gamma_s])

    gamma_all = np.delete(gamma_all, 0, 0)
    vf.confirm_0(gamma_all.shape - np.array([npos*nz/6, 2]) )
    
    filename = 'SRO_gamma_all_%d.txt' %(b_slip)
    np.savetxt(filename, gamma_all)
    os.remove('tmp_filelist')


    write_output(gamma_all, npos, nz, a_fcc, b_slip)
    plot_hist(b_slip)

    return gamma_all.mean(), gamma_all.std() 





# for each structure

def calc_gamma_s(EPI_beta, pos_filename, a_fcc, b_slip):

    atoms = vf.my_read_vasp( pos_filename )
    latt = atoms.get_cell()[:]
    natoms = atoms.get_positions().shape[0]

    b = atoms.pos_a0/np.sqrt(2)
    nx = np.around( latt[0,0]/b )
    vf.confirm_int(nx)
    nx = int(nx)
        
    nlayers, nmiss = vf_shift.check_layers(filename = pos_filename)
    vf.confirm_0(nmiss)
    nz = nlayers

    E_s = calc_E_s(atoms, nx, nz, EPI_beta, b_slip)
    
    vf.confirm_0( E_s.shape - np.array([nz/6, 2]) )

    qe = vf.phy_const('qe')
    gamma_s = E_s/(natoms/nz) / (np.sqrt(3)/2*a_fcc**2/2) *qe*1e23

    return gamma_s, nz







# for each structure, sum of nz planes

def calc_E_s(atoms_in, nx, nz, EPI_beta, b_slip):
    atoms = copy.deepcopy(atoms_in)
    natoms = atoms.get_positions().shape[0]

    latt = atoms.get_cell()[:]
    dz = latt[2,2]/nz

    E0 = vmc.eval_Ef_from_EPI(atoms, EPI_beta) 

    E_s = np.zeros( 2 )  # + and -
    
    for i in np.arange( nz/6 ):
        atoms2 = copy.deepcopy(atoms)
        
        pos2 = atoms2.get_positions()
        pos2[:,2] = pos2[:,2] + i* dz*6
        atoms2.set_positions(pos2, apply_constraint=False )
        atoms2.wrap()
            
        E_p = calc_E_p(atoms2, nx, EPI_beta, b_slip)
        E_s = np.vstack([E_s, E_p])

    E_s = np.delete(E_s, 0, 0)
    E_s = (E_s - E0)*natoms

    return E_s






# for each plane, + and - 

def calc_E_p(atoms_in, nx, EPI_beta, b_slip):
    atoms2 = copy.deepcopy(atoms_in)
    latt2 = atoms2.get_cell()[:]
    dx = latt2[0,0]/nx

    E_p = np.array([])
    for i in np.array([-1, 1]) *b_slip :
        atoms3 = copy.deepcopy(atoms2)
       
        latt3 = atoms3.get_cell()[:]
        latt3[2,0] = latt3[2,0] + i*dx
        atoms3.set_cell( latt3 )
        atoms3.wrap()

        temp = vmc.eval_Ef_from_EPI(atoms3, EPI_beta) 
        E_p = np.append(E_p, temp)
    
    return E_p 








def write_output(gamma_all, npos, nz, a_fcc, b_slip):
    
    filename = 'SRO_tau_A_%d.txt' %(b_slip)
    f = open(filename, 'w+')
        
    f.write('# SRO average strengthening: \n' )


    f.write('%16s %16s %16s \n' \
        %('npos', 'nz', 'a_fcc (Ang)') )
    f.write('%16d %16d %16.8f \n\n' \
        %(npos, nz, a_fcc) )

   
    f.write('# gamma_all (mJ/m^2): \n' )
    f.write('%16s %16s \n' \
        %('mean', 'std') )
    f.write('%16.8f %16.8f \n\n' \
        %(  gamma_all.mean(), \
            gamma_all.std() ))


    f.write('# tau (MPa): \n' )
    f.write('%16s %16s \n' \
        %('mean', 'std') )
    f.write('%16.8f %16.8f \n\n' \
        %( calc_tau_from_gamma( a_fcc, gamma_all.mean() ), \
           calc_tau_from_gamma( a_fcc, gamma_all.std()  ) ))




def calc_tau_from_gamma(a_fcc, gamma):
    tau =  gamma/(a_fcc/np.sqrt(2)) *1e-3*1e10 *1e-6
    return tau 


 




def plot_hist(b_slip):

    filename='SRO_gamma_all_%d.txt' %(b_slip)
    gamma_all =  np.loadtxt( filename )

    miu = np.mean(gamma_all)
    sigma = np.std(gamma_all)


    fig_wh = [3.15, 2.8]
    fig_subp = [1, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)

    ax1.set_position([0.2, 0.15, 0.75, 0.8])

    ax1.hist(x=gamma_all, bins=20, density=True) 

    ax1.set_xlabel('$\\gamma_\\mathrm{APB}$ (mJ/m$^2$)')
    ax1.set_ylabel('Probability density')
    
    xmin, xmax = ax1.get_xlim()
    ax1.set_xlim([xmin, xmax] )
    
    ymin, ymax = ax1.get_ylim()

    xi = np.linspace( xmin, xmax ) 
    yi = my_gaussian(miu, sigma, xi)

    ax1.plot(xi, yi, '--k')

    str1 = 'mean = %.3f\nstd = %.3f' %(miu, sigma )
    ax1.text( xmin+(xmax-xmin)*0.95, ymin+(ymax-ymin)*0.8, \
        str1 , horizontalalignment='right'  )


    filename = 'SRO_gamma_hist_%d.pdf' %(b_slip)
    plt.savefig(filename)
    plt.close('all')





def my_gaussian(miu, sigma, x):
    g = 1/np.sqrt(2 * np.pi ) / sigma  \
        * np.exp(- (x-miu)**2 / (2*sigma**2) )
    return g








# plot_gamma_slip()


