#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy
from myvasp import vasp_func as vf 






def bestsqs_to_POSCAR(filename='bestsqs-1.out'):

    with open(filename) as f:
        sqs = f.read().splitlines()


    latt = np.zeros([3, 3])
    for i in np.arange(3):
        temp = sqs[3+i].split(' ')

        for j in np.arange(3):
            latt[i, j] = float( temp[j] )


    natoms = len(sqs)-6
    pos = np.zeros([natoms, 3])
    lelem = np.zeros( natoms )

    for i in np.arange(natoms):
        temp = sqs[6+i].split(' ')

        for j in np.arange(3):
            pos[i, j] = float( temp[j] )
    
        s = temp[3]
        if s == 'A':
            t = 1
        elif s == 'B':
            t = 2
        elif s == 'C':
            t = 3
        elif s == 'D':
            t = 4
        elif s == 'E':
            t = 5
        elif s == 'F':
            t = 6
        elif s == 'G':
            t = 7
        lelem[i] = t 
    
    temp = lelem[:].argsort()
    pos   =   pos[ temp, :]
    lelem = lelem[ temp   ]

    temp = 'POSCAR_'+filename[8]
    pos_a0 = 4.0
    write_poscar(pos_a0, latt*pos_a0, lelem, pos*pos_a0, filename=temp)







def write_poscar(pos_a0, latt, lelem, pos, filename='POSCAR'):

    latt = latt/pos_a0 
    pos  = pos/pos_a0 

    temp = lelem.max()
    vf.confirm_int(temp)
    ns = np.zeros(int(temp)) 

    for i in np.arange( len(ns) ):
        mask = lelem[:]==(i+1) 
        temp = lelem[mask]
        ns[i] = temp.shape[0]

    mask = ns[:] != 0
    ns = ns[mask]

    f = open(filename, 'w+')
    f.write('system name \n %22.16f \n' %(pos_a0) )

    for i in np.arange(3):
        f.write(' %22.16f %22.16f %22.16f \n' %(latt[i,0], latt[i,1], latt[i,2])  )

    for i in np.arange(len(ns)):
        f.write(' %d ' %(ns[i]) )
    f.write('\nS \nC \n')

    for i in np.arange(pos.shape[0]):
        f.write(' %22.16f %22.16f %22.16f   T T T \n' %(pos[i,0], pos[i,1], pos[i,2])  )

    f.close() 

   






def get_list_of_outcar():
    from ase.io.vasp import read_vasp_out

    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    latoms2 = []   # list of ASE_Atoms from OUTCAR
    for i in jobn:
        filename = './y_dir/%s/OUTCAR' %(i)
        atoms2 = read_vasp_out(filename)
        latoms2.append(atoms2)
    return latoms2




def get_list_of_atoms():
    jobn, Etot, Eent, pres = vf.vasp_read_post_data()
    latoms = []   # list of ASE_Atoms from CONTCAR
    
    os.chdir('y_dir')
    for i in np.arange( len(jobn) ):
        os.chdir( jobn[i] )
        atoms = my_read_vasp('CONTCAR')
        latoms.append(atoms)
        os.chdir('..')
    os.chdir('..')

    return latoms






def get_list_of_atoms_from_poscars2(dirname='poscars2'):
    os.system('ls %s/  >  tmp_filelist' %(dirname) )
    latoms = []   # list of ASE_Atoms from CONTCAR

    f = open('tmp_filelist', 'r')
    for line in f:
        atoms = my_read_vasp( line.strip('\n') )
        latoms.append(atoms)
    f.close() 
    
    os.remove('tmp_filelist')
    return latoms







#==============================


def my_read_vasp(filename):
    from ase.io.vasp import read_vasp 
    import types 
    
    atoms = read_vasp(filename)
    with open(filename, 'r') as f:
        atoms.pos_a0 = float( f.readlines()[1] )

    atoms.get_cn    = types.MethodType(get_cn,    atoms) 
    atoms.get_nelem = types.MethodType(get_nelem, atoms) 

    return atoms




def get_cn(atoms):
    import pandas as pd

    natoms = atoms.get_positions().shape[0]
    atoms_an = atoms.get_atomic_numbers()

    # number of atoms of each element
    natoms_elem = np.array([])
    for i in pd.unique(atoms_an):
        mask = np.isin(atoms_an, i)
        natoms_elem = np.append( natoms_elem, \
            atoms_an[mask].shape[0] )

    if np.abs( natoms - natoms_elem.sum() ) > 1e-10:
        sys.exit('==> ABORT. wrong natoms_elem. ')
    cn = natoms_elem / natoms
    return cn





def get_nelem(atoms):
    import pandas as pd
    atoms_an = atoms.get_atomic_numbers()
    nelem = len( pd.unique(atoms_an) ) 
    return nelem 
    








def my_write_vasp(atoms_in, filename='POSCAR', vasp5=True):
    from ase.io.vasp import write_vasp

    atoms = copy.deepcopy(atoms_in)
    pos_a0 = atoms.pos_a0

    atoms.set_cell( atoms.get_cell()/pos_a0 )
    atoms.set_positions( atoms.get_positions()/pos_a0, \
        apply_constraint=False ) 
    
    write_vasp('POSCAR_temp', atoms,
    label='system_name', direct=False, vasp5=vasp5)

    with open('POSCAR_temp', 'r') as f:
        lines = f.readlines()
    lines[1] = ' %.16f \n' % (pos_a0)

    with open(filename, "w") as f:
        f.writelines(lines)
    os.remove('POSCAR_temp')




#================================================


def my_read_doscar(fname="DOSCAR"):
    from scipy import integrate

    # Read a VASP DOSCAR file
    f = open(fname)
    natoms = int(f.readline().split()[0])
    [f.readline() for nn in range(4)]  # Skip next 4 lines.

    # First we have a block with total and total integrated DOS
    line = f.readline().split()
    Emax   = float( line[0] )
    Emin   = float( line[1] )
    ndos   =   int( line[2] )
    Efermi = float( line[3] )

    tdos_all = vf.my_read_line(f)
    for nd in np.arange(1, ndos):
        tdos_all = np.vstack([tdos_all, vf.my_read_line(f)])

    # Next we have one block per atom, if INCAR contains the stuff
    # necessary for generating site-projected DOS
    lpdos = []    # list of pdos     
    for na in np.arange(natoms):
        line = f.readline().split()
        if len(line) == 0:
            break     # No site-projected DOS            
        else:
            vf.confirm_0( float( line[0] ) - Emax ) 
            vf.confirm_0( float( line[1] ) - Emin ) 
            vf.confirm_0( float( line[2] ) - ndos ) 
            vf.confirm_0( float( line[3] ) - Efermi ) 

        pdos = vf.my_read_line(f)
        for nd in np.arange(1, ndos):
            pdos = np.vstack([pdos, vf.my_read_line(f)])   
        lpdos.append(pdos)   

    #------------------------

    Ei = np.linspace(Emin, Emax, ndos)  - Efermi 
    dE = (Emax - Emin)/(ndos-1)
    vf.confirm_0( np.diff(Ei)[0] - dE ) 

    if tdos_all.shape[1] == 3:
        tdos = tdos_all[:,1].copy()
        idos = tdos_all[:,2].copy()

        tdos  = tdos[:, np.newaxis]
        idos  = idos[:, np.newaxis]

    elif tdos_all.shape[1] == 5:
        tdos = tdos_all[:,1:3].copy()        
        idos = tdos_all[:,3:5].copy()

        tdos[:,1] = tdos[:,1]*(-1)
        idos[:,1] = idos[:,1]*(-1)

    else:
        sys.exit('ABORT: wrong tdos_all.')
    
    # to compare with the output idos 
    idos2 = integrate.cumulative_trapezoid( tdos, Ei, axis=0, initial=0)

    return atoms_dos(Ei, tdos, idos, idos2, lpdos)  



class atoms_dos:
    def __init__(self, Ei, tdos, idos, idos2, lpdos):
        self.Ei = Ei
        self.tdos = tdos
        self.idos = idos
        self.idos2 = idos2
        self.lpdos = lpdos 


    def plot_dos(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
         
        Ei    = self.Ei 
        tdos  = self.tdos.copy()
        idos  = self.idos.copy()
        idos2 = self.idos2.copy()
     
        fig_wh = [3, 4.5]
        fig_subp = [2, 1]
        fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
     
        fig_pos  = np.array([0.25, 0.55, 0.7, 0.4])
        fig_dpos = np.array([0, -0.45, 0, 0])        
        for i in np.arange(2):    
            ax1[i].set_position(fig_pos + fig_dpos*i)
    
        lc=['C0', 'C1']
        for j in np.arange( tdos.shape[1] ):     
            ax1[0].plot( Ei , tdos[:,j],  '-',   color=lc[j] )
            ax1[1].plot( Ei , idos[:,j],  '-',   color=lc[j] )
            ax1[1].plot( Ei , idos2[:,j], '--',  color=lc[j] )
     
        if idos.shape[1]==2:
            ax1[1].plot( Ei , idos[:,0] - idos[:,1] , '--', color='k', linewidth=0.5 )
            ax1[1].plot( Ei , idos[:,0] + idos[:,1] , '--', color='k', linewidth=0.5 )

        fig_xlim = [Ei.min(), Ei.max()]
        ax1[0].set_xlim( fig_xlim )
        for i in np.arange(2):
            ax1[i].plot( fig_xlim, [0, 0], '--', color='gray' )
            
            fig_ylim = ax1[i].get_ylim()
            ax1[i].set_ylim( fig_ylim )
            ax1[i].plot( [0, 0], fig_ylim, '--', color='gray' )
     
        if idos.shape[1]==1:
            ne     = vf.my_interp(Ei, idos[:,0], 0)
            nmag   = 0 
            nbands = vf.my_interp(Ei, idos[:,0], Ei.max())     
        elif idos.shape[1]==2:
            ne     = vf.my_interp(Ei, idos[:,0] - idos[:,1], 0)
            nmag   = vf.my_interp(Ei, idos[:,0] + idos[:,1], 0) 
            nbands = vf.my_interp(Ei, idos[:,0] - idos[:,1], Ei.max())     
        else:
            sys.exit('ABORT: wrong idos.') 
                
        str1 = 'NELECT=%.3f\nNmag=%.3f\nNBANDS*2=%.3f'   %(ne, nmag, nbands) 
        vf.my_text(ax1[1], str1, 0.03, 0.78)

        ax1[0].set_ylabel('DOS (e/eV)')
        ax1[1].set_ylabel('Integrated DOS (e)')
        ax1[1].set_xlabel('$E - E_\\mathrm{Fermi}$ (eV)')
    
        filename = 'y_post_dos.pdf'
        plt.savefig(filename)
        plt.close('all')









