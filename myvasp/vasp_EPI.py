
import numpy as np 
from myvasp import vasp_func as vf 
import copy, os, sys, shutil, time



#===================
# MC
#===================

def MC_swap_with_EPI(atoms_in, EPI_beta, T, nstep):
    
    jobname = 'T_%.0f_nstep_%d' %(T, nstep)


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


        if atom_num[sid1] == atom_num[sid2]:    
            # the same element
            Ef_new = Ef_all[-1]

        else:
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

        if np.mod(i, dstep) == 0:
            write_MC_poscar(atoms, Ef_new -EPI_beta[0], istep=i)

    # MC loop ends
    #==================================


          

    filename = 'Ef_all_%s' %(jobname)
    np.savetxt(filename, Ef_all )
   
    plot_MC(EPI_beta, Ef_all, T)

    move_file_to_dump('POSCAR*', dumpdir)
    move_file_to_dump('Ef_all*', dumpdir)
    move_file_to_dump('fig_MC*', dumpdir)

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
    from myvasp import vasp_calc_pairs_per_shell as vf2

    atoms = copy.deepcopy( atoms_in )
    nelem = len( atoms.cn )

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




def move_file_to_dump(files, dumpdir):
    import glob  
    f = glob.iglob(files)  
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

    filename = 'fig_MC_T_%.0f_nstep_%d.pdf' %(T, len(Ef_all)-1)
    plt.savefig(filename)
    plt.close('all')



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





# run_MC_case(nstep=1000, T_list=[3000.0])







