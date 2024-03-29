
import numpy as np 
from myvasp import vasp_func as vf 
import copy, os, sys, shutil, time



#===================
# MC
#===================

def MC_swap_with_EPI(atoms_in, EPI_beta, T, nstep):
    
    jobname = 'T_%04.0f_nstep_%d' %(T, nstep)


    # mkdir dump with backup
    dumpdir = 'dump_%s' %(jobname)
    try:
        os.mkdir(dumpdir)
    except:
        str1 = time.strftime('%Y%m%d_%H%M%S', time.localtime()) 
        str2 = '%s_%s' %(dumpdir, str1) 
        os.rename(dumpdir, str2)
        os.mkdir(dumpdir)
    
    
    
    atoms = copy.deepcopy(atoms_in)
    natoms = len( atoms.get_positions() )
    atom_num = atoms.get_atomic_numbers()

    kB = 8.617333e-5  #[eV/K]	
    kT = kB*T
    beta2 = natoms/kT 

    dstep = calc_dstep_from_nstep(nstep)
    

   

    # initialization
    Ef0 = eval_Ef_from_EPI(atoms, EPI_beta) 
    write_MC_poscar(atoms, Ef0 -EPI_beta[0])
    Ef_all = np.array([Ef0]) 


    Ef_write_pos = EPI_beta[0]+0.5
    while Ef_write_pos > Ef0 :  
        Ef_write_pos -= 0.01




    #==================================
    # enter MC loop
    for i in np.arange(1, nstep+1):
        print('==> MC STEP:', i)
    
        
        sid1 = rand_id(natoms)
        sid2 = rand_id(natoms)

        while atom_num[sid1] == atom_num[sid2]:    
            # the same element
            sid1 = rand_id(natoms)
            sid2 = rand_id(natoms)

        
        # different element
        pos = atoms.get_positions()
        temp = pos[sid1,:].copy()
        pos[sid1,:] = pos[sid2,:].copy()
        pos[sid2,:] = temp.copy()
        
        atoms2 = copy.deepcopy(atoms)
        atoms2.set_positions(pos, apply_constraint=False )
        Ef2 = eval_Ef_from_EPI(atoms2, EPI_beta) 
        dEf = Ef2 - Ef_all[-1]    
        
        
        #acceptance probability
        P = np.exp( -dEf * beta2 )
        print('dEf, P:', dEf, P)
        if P > np.random.random_sample() :
            atoms = copy.deepcopy(atoms2)
            Ef_new = Ef2
        else:
            Ef_new = Ef_all[-1]
        
        Ef_all = np.append(Ef_all, Ef_new)


        # write pos
        if Ef_new < Ef_write_pos:
            write_MC_poscar(atoms, Ef_new -EPI_beta[0])
            Ef_write_pos -= 0.01
        
        if i==1 or i==nstep :  
            write_MC_poscar(atoms, Ef_new -EPI_beta[0])

        if (i>nstep*0.4) and (np.mod(i, dstep) == 0):
            write_MC_poscar(atoms, Ef_new -EPI_beta[0], istep=i)

    # MC loop ends
    #==================================


          

    filename = 'Ef_all_%s' %(jobname)
    np.savetxt(filename, Ef_all )
   
    plot_MC(EPI_beta, Ef_all, T)

    move_file_to_dump('POSCAR*', dumpdir)
    move_file_to_dump('Ef_all*', dumpdir)
    move_file_to_dump('fig_MC*', dumpdir)


    os.chdir(dumpdir)
    analyze_dump(jobname)
    os.chdir('..')
    
    return Ef_all









#======================

def write_MC_poscar(atoms_in, Ef, istep=0):
    atoms = copy.deepcopy(atoms_in)
    if istep > 0.5:
        pos_name = 'POSCAR_step_%d_Ef_%+.3f' %(istep, Ef)
    else:
        pos_name = 'POSCAR_Ef_%+.3f' %(Ef)
    vf.my_write_vasp(atoms, filename=pos_name, vasp5=True)

     


def rand_id(natoms):
    y = int( np.floor( np.random.random_sample()*natoms ) )
    return y




def eval_Ef_from_EPI(atoms_in, EPI_beta):
    from myvasp import vasp_EPI_dp_shell as vf2

    atoms = copy.deepcopy( atoms_in )
    nelem = atoms.get_nelem()

    shellmax = (len(EPI_beta)-1)/ (nelem*(nelem-1)/2) 
    vf.confirm_int(shellmax)
    shellmax = int(shellmax)

    dp_shell = vf2.calc_pairs_per_shell(atoms, shellmax=shellmax, write_dp=False)
    X = np.append(1.0, -0.5* dp_shell)
    Ef = np.dot(X, EPI_beta) 
    return Ef 




def calc_dstep_from_nstep(nstep):
    t1 = nstep/1000
    t2 = nstep/100
    dstep = 1    # dump every dstep
    while not ( dstep > t1 and dstep <= t2 ):
        dstep = int(dstep *10)
        if dstep > nstep:
            dstep = int(nstep/2)   # nstep is too small 
            break
    print('==> dump every dstep:', dstep)
    return dstep




def move_file_to_dump(filename, dumpdir):
    import glob  
    f = glob.iglob(filename)  
    for i in f:  
        shutil.move(i, dumpdir)

   




def plot_MC(EPI_beta, Ef_all, T):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Ef_rand = EPI_beta[0]
   
    fig_wh = [3.15, 2.7]
    fig_subp = [1, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
    ax1.set_position([0.23, 0.16, 0.72, 0.78])

    xi = np.arange( len(Ef_all) ) / 1e3
    ax1.plot(xi, (Ef_all-Ef_rand)*1e3,  '-' , \
        label='$T_a$ = %d K' %(T))
   

    ax1.set_xlabel('MC step ($\\times 10^3$)')
    ax1.set_ylabel('$E_{f,\\mathrm{Pred}} - E^\\mathrm{rand}_{f}$ (meV/atom)')
    ax1.legend( framealpha=0.5, loc="best")

    filename = 'fig_MC_T_%04.0f_nstep_%d.pdf' %(T, len(Ef_all)-1)
    plt.savefig(filename)
    plt.close('all')





def calc_shellmax(atoms_in, EPI_beta):
    atoms = copy.deepcopy(atoms_in)
    nelem = atoms.get_nelem()
    shellmax = (len(EPI_beta)-1)/ (nelem*(nelem-1)/2) 
    vf.confirm_int(shellmax)
    shellmax = int(shellmax)
    return shellmax 







def analyze_dump(jobname):

    atoms = vf.my_read_vasp('../CONTCAR')
    nelem = atoms.get_nelem()

    EPI_beta =  np.loadtxt('../y_post_EPI.beta_4.txt')

    # extend EPIs to 10 shells 
    if nelem == 2:
        EPI_beta = np.append(EPI_beta, np.zeros(11-len(EPI_beta)) )  

    else:
        shellmax = calc_shellmax( atoms, EPI_beta )

        temp = int( (len(EPI_beta)-1) /shellmax * (10-shellmax) )
        temp2 = np.zeros( temp )

        EPI_beta = np.append( EPI_beta, temp2 )

    print('==> extended EPI_beta:')
    print( EPI_beta )



    os.system('ls POSCAR_step_* > tmp_filelist')
    f = open('tmp_filelist', 'r')

    dp_shell_tot = np.zeros(len(EPI_beta)-1)
    WC_SRO_tot   = np.zeros(len(EPI_beta)-1)

    for line in f:
        filename = line.strip('\n') 
        print(filename)
        atoms = vf.my_read_vasp(filename)
   
        dp_shell = plot_dp_shell(atoms, EPI_beta=EPI_beta)
        dp_shell_tot = np.vstack([dp_shell_tot, dp_shell])

        filename2 = 'y_post_dp_shell_%s.pdf' %(filename)
        os.rename(  'y_post_dp_shell.pdf', filename2)


        os.remove('y_post_dp_shell.txt')

        filename2 = 'y_post_WC_SRO_shell.txt' 
        temp = np.loadtxt(filename2)
        WC_SRO_tot = np.vstack([WC_SRO_tot, temp])
        os.remove(filename2)

    os.remove('tmp_filelist')

    dp_shell_tot = np.delete(dp_shell_tot, 0, 0)
    print('==> dp_shell_tot.shape[0]:', dp_shell_tot.shape[0])
    
    dp_shell_avg = np.mean(dp_shell_tot, axis=0)
    filename2 = 'y_post_dp_shell_avg_%s.txt' %(jobname)
    np.savetxt(filename2, dp_shell_avg )

    plot_dp_shell(atoms, EPI_beta, dp_shell=dp_shell_avg)
    filename2 = 'fig_dp_shell_avg_%s.pdf' %(jobname)
    os.rename('y_post_dp_shell.pdf', filename2)


    WC_SRO_tot = np.delete(WC_SRO_tot, 0, 0)
    print('==> WC_SRO_tot.shape[0]:', WC_SRO_tot.shape[0])

    WC_SRO_avg = np.mean(WC_SRO_tot, axis=0)
    filename2 = 'y_post_WC_SRO_shell_avg_%s.txt' %(jobname)
    np.savetxt(filename2, WC_SRO_avg )

    plot_dp_shell(atoms, EPI_beta, dp_shell=WC_SRO_avg, dp_type='SRO')
    filename2 = 'fig_WC_SRO_shell_avg_%s.pdf' %(jobname)
    os.rename('y_post_WC_SRO_shell.pdf', filename2)




def plot_dp_shell(atoms_in, EPI_beta=np.array([]), dp_shell=np.array([]), dp_type='dp' ) :
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd 
    from myvasp import vasp_EPI_dp_shell as vf2


    atoms = copy.deepcopy(atoms_in)

    nelem = atoms.get_nelem()
    shellmax = (len(EPI_beta)-1)/ (nelem*(nelem-1)/2) 
    vf.confirm_int(shellmax)
    shellmax = int(shellmax)

    if len(dp_shell)<0.1 :
        dp_shell = vf2.calc_pairs_per_shell(atoms, \
            shellmax=shellmax, write_dp=True)
    else:
        print('==> Use input dp_shell.')

    temp = int(len(dp_shell)/shellmax)
    dp_shell_2 = dp_shell.reshape(shellmax, temp)

    rcry, ncry = vf2.crystal_shell('fcc')
    xi = rcry[0:shellmax].copy()


    fig_xlim = np.array([ xi[0]-0.15, xi[-1]+0.15 ])
    # fig_ylim = np.array([-0.5, 0.5])

    elem_sym = pd.unique( atoms.get_chemical_symbols() )
    print('elem_sym:', elem_sym)
    nelem = elem_sym.shape[0]

    fig_wh = [2.7, 2.5]
    fig_subp = [1, 1]            
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)

    fig_pos  = np.array([0.27, 0.17, 0.70, 0.78])
    ax1.set_position(fig_pos)

    ax1.plot( fig_xlim, [0, 0], ':k' )
  

    k=-1
    for i in np.arange(nelem):
        for j in np.arange(i, nelem):
            if i != j:
                k = k+1

                elems_name = '%s%s' %( elem_sym[i], elem_sym[j] )
                # str1 = '$\\Delta \\eta_{\\mathrm{%s}, d}$' %(elems_name)
                str1 = '%s' %(elems_name)
                mycolor, mymarker =  mycolors(elems_name)
                               
                ax1.plot(xi,  dp_shell_2[:, k], '-', \
                    color = mycolor, marker = mymarker, \
                    label = str1)
      

    vf.confirm_0( dp_shell_2.shape[1] - (k+1) ) 
    ax1.legend(loc='best', ncol=2, framealpha=0.4, \
        fontsize=7)

    ax1.set_xlabel('Pair distance $d/a$')
    
    ax1.set_xlim( fig_xlim )
    # ax1.set_ylim( fig_ylim )
    fig_ylim = ax1.get_ylim()


    if dp_type == 'dp':
        ax1.set_ylabel('$\\Delta \\eta_{nm, d}$')
    
        if len(EPI_beta) > 0.1 :
            Ef = -0.5 * np.dot( dp_shell, EPI_beta[1:])
    
            str1 = '$E_{f, \\mathrm{Pred}} - {E}^\\mathrm{rand}_{f}$\n= %.3f meV/atom' %( Ef*1e3 )
            ax1.text( 
                fig_xlim[0]+(fig_xlim[1]-fig_xlim[0])*0.95, \
                fig_ylim[0]+(fig_ylim[1]-fig_ylim[0])*0.06, str1, \
                horizontalalignment='right' )
    
        filename = 'y_post_dp_shell.pdf'


    elif dp_type == 'SRO':
        ax1.set_ylabel('Warren-Cowley SRO $\\alpha_{nm, d}$')
             
        filename = 'y_post_WC_SRO_shell.pdf'


    plt.savefig(filename)
    plt.close('all')
    return dp_shell





def mycolors(elems_name):
    d={}
    d['AuNi'] = ['C0', 'o']      
    d['AuPd'] = ['C1', 's']    
    d['AuPt'] = ['C2', '^']      
    d['NiPd'] = ['C3', 'v']      
    d['NiPt'] = ['C4', '<']      
    d['PdPt'] = ['C5', '>']   

    d['AuCu'] = ['C6', 'P']   
    d['NiCu'] = ['C7', 'X']   
    d['PdCu'] = ['C8', 'p']   
    d['PtCu'] = ['C9', 'H']   
 
    d['AuAg'] = [ np.array([3,168,158])/255,   'o']   
    d['AgCu'] = [ np.array([240,128,128])/255, 's']   

    d['NiCo'] = ['C0', 'o']   
    d['NiCr'] = ['C1', 's']   
    d['CoCr'] = ['C2', '^']   


    if elems_name in d:
        y1 = d[elems_name][0]
        y2 = d[elems_name][1]
    else:
        y1 = 'k'
        y2 = 'o'
    return y1, y2











#============================
# run MC case

def run_MC_case(nstep=1000, T_list=[300.0, 1500.0], \
    EPI_filename='y_post_EPI.beta_4.txt', \
    pos_filename='CONTCAR'):

    EPI_beta =  np.loadtxt( EPI_filename )
    atoms = vf.my_read_vasp( pos_filename )
    for T in np.array(T_list):
        print('==> MC starts, with T:', T)
        MC_swap_with_EPI(atoms, EPI_beta, T, nstep)
   
#============================





# run_MC_case(nstep=1000, T_list=[1000.0])



