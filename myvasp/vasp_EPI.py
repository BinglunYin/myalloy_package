
import numpy as np 
from myvasp import vasp_func as vf 
from myvasp import vasp_calc_pairs_per_shell as vf2 
import copy, os, sys




#===================
# MC
#===================

def MC_swap_with_EPI(atoms_in, EPI_beta, T, nstep):
    atoms = copy.deepcopy(atoms_in)
    
    kB = 8.617333e-5  #[eV/K]	
    kT = kB*T

    natoms = len( atoms.get_positions() )
    atom_num = atoms.get_atomic_numbers()

    Ef0 = eval_Ef_from_EPI(atoms, EPI_beta) 
    Ef_all=np.array([Ef0]) 

    Ef_write_pos = EPI_beta[0]+0.1

    while Ef_write_pos > Ef0 :  
        Ef_write_pos -= 0.01


    for i in np.arange(1, nstep+1):
        print('==> step ', i)
        
        sid1 = rand_id(natoms)
        sid2 = rand_id(natoms)

        while atom_num[sid1] == atom_num[sid2]:
            sid2 = rand_id(natoms)

        pos = atoms.get_positions()
        temp = pos[sid1,:].copy()
        pos[sid1,:] = pos[sid2,:].copy()
        pos[sid2,:] = temp.copy()

        atoms2 = copy.deepcopy(atoms)
        atoms2.set_positions(pos, apply_constraint=False )
        Ef2 = eval_Ef_from_EPI(atoms2, EPI_beta) 
        dEf = Ef2 - Ef_all[-1]    

        #acceptance probability
        P = np.exp(- dEf * natoms /kT)
        print('dEf, P:', dEf, P)

        if P > np.random.random_sample() :
            atoms = copy.deepcopy(atoms2)
            temp = Ef2
        else:
            temp = Ef_all[-1]
        
        Ef_all = np.append(Ef_all, temp)

        if temp < Ef_write_pos:
            write_MC_poscar(atoms, Ef_all[i] -EPI_beta[0])
            Ef_write_pos -= 0.01
        
        if i==1 or i==nstep :  
            write_MC_poscar(atoms, Ef_all[i] -EPI_beta[0])

          

    filename = 'Ef_all_T_%.0f_nstep_%d.txt' %(T, nstep)
    np.savetxt(filename, Ef_all )
   
    return Ef_all




def write_MC_poscar(atoms_in, Ef):
    atoms = copy.deepcopy(atoms_in)
    pos_name = 'POSCAR_Ef_%+.3f' %(Ef)
    vf.my_write_vasp(atoms, filename=pos_name, vasp5=True)
          



def eval_Ef_from_EPI(atoms_in, EPI_beta):
    atoms = copy.deepcopy( atoms_in )
    nelem = len( atoms.cn )

    shellmax = (len(EPI_beta)-1)/ (nelem*(nelem-1)/2) 
    vf.confirm_int(shellmax)
    shellmax = int(shellmax)

    dp_shell = vf2.calc_pairs_per_shell(atoms, shellmax=shellmax, write_dp=False)
    X = np.append(1.0, -0.5* dp_shell)
    Ef = np.dot(X, EPI_beta) 
    return Ef 




def rand_id(natoms):
    y = int( np.ceil( np.random.random_sample()*natoms ) -1)
    return y



