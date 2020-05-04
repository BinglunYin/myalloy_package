#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy
import matplotlib.pyplot as plt
from ase.io.vasp import read_vasp, read_vasp_out, write_vasp
from ase import Atoms




def create_random_alloys(atoms_in, cn, nsamples=1, id1=1, vasp5=False):
    atoms = copy.deepcopy(atoms_in)
    natoms = atoms.get_positions().shape[0]
   
    # calc natoms_elem
    cn = cn/cn.sum()     
    natoms_elem = np.around( cn * natoms )

    if natoms_elem.min() < 0.1:
        sys.exit('==> ABORT. natoms is too small. ')

    max_elem = np.argmax( natoms_elem )
    temp = natoms_elem.sum() - natoms_elem[max_elem]
    natoms_elem[max_elem] = natoms - temp
    print(natoms_elem)

    # new symbols
    symb = np.array([])
    for i in np.arange(natoms_elem.shape[0]):
        for j in np.arange(natoms_elem[i]):
            symb = np.append(symb, i+1)
    atoms.set_chemical_symbols( symb )

    # randomize pos
    for i in np.arange(nsamples):
        temp = np.hstack([ atoms.positions, \
            np.random.random_sample([natoms, 1]) ])
        ind = np.argsort(temp[:, -1])
        atoms.set_positions(temp[ind, 0:3], apply_constraint=False)
  
        filename = 'POSCAR_%03d' %( i + id1 )
        my_write_vasp(atoms, filename, vasp5=vasp5)




# class my_Atoms(Atoms):
#     def __init__(self, a0=1.0, *args, **kwargs):
#         self.pos_a0 = a0      # a0 in the POSCAR
#         super().__init__(*args, **kwargs)
   
#     def set_pos_a0(self, pos_a0):
#         self.pos_a0 = pos_a0
#     def get_pos_a0(self):
#         return self.pos_a0



def my_read_vasp(filename):
    atoms = read_vasp(filename)
    with open(filename, 'r') as f:
        atoms.pos_a0 = float( f.readlines()[1] )
    return atoms



def my_write_vasp(atoms_in, filename='POSCAR', vasp5=True):
    atoms = copy.deepcopy(atoms_in)
    pos_a0 = atoms.pos_a0

    atoms.set_cell( atoms.get_cell()/pos_a0 )
    atoms.set_positions( atoms.get_positions()/pos_a0, apply_constraint=False ) 
    
    write_vasp('POSCAR_temp', atoms,
    label='system_name', direct=False, vasp5=vasp5)

    with open('POSCAR_temp', 'r') as f:
        lines = f.readlines()
    lines[1] = ' %.8f \n' % (pos_a0)

    with open(filename, "w") as f:
        f.writelines(lines)
    os.remove('POSCAR_temp')





def mylinreg(X, y):
    if X.shape[0] < X.shape[1]:
        sys.exit('==> ABORT. too few data. ', \
            X.shape[0], X.shape[1])

    temp = np.linalg.matrix_rank(X)
    if temp < X.shape[1]:
        sys.exit('==> ABORT. linear regression ill-conditioned. ', \
            temp, X.shape[1])

    beta = np.linalg.inv(X.T @ X) @ X.T @ y 
    SStot = np.sum( (y-y.mean())**2 )
    SSres = np.sum( (y-X@beta)**2 )
    R2 = 1- SSres/SStot
    print('==> beta, R2:', beta, R2)
    return beta, R2




def run_cmd_in_jobn(mycmd, **kwargs):
    print('==> mycmd, args:', mycmd, kwargs)
    jobn, Etot, Eent, pres = vasp_read_post_data()
    os.chdir('y_dir')
    print( os.getcwd() )
    for i in np.arange( len(jobn) ):
        os.chdir( jobn[i] )
        print( os.getcwd() )
        mycmd(**kwargs)
        os.chdir('..')
    os.chdir('..')
    print( os.getcwd() )




def get_list_of_outcar():
    jobn, Etot, Eent, pres = vasp_read_post_data()
    latoms2 = []   # list of ASE_Atoms from OUTCAR
    for i in jobn:
        filename = './y_dir/%s/OUTCAR' %(i)
        atoms2 = read_vasp_out(filename)
        latoms2.append(atoms2)
    return latoms2



def get_list_of_atoms():
    jobn, Etot, Eent, pres = vasp_read_post_data()
    latoms = []   # list of ASE_Atoms
    for i in jobn:
        filename = './y_dir/%s/CONTCAR' %(i)
        ASE_Atoms = read_vasp(filename)
        latoms.append(ASE_Atoms)
    return latoms


def my_plot(fig_wh, fig_subp):
    plt.rcParams['font.size']=8
    #plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.linewidth']=0.5
    plt.rcParams['axes.grid']=True
    plt.rcParams['grid.linestyle']='--'
    plt.rcParams['grid.linewidth']=0.2
    plt.rcParams["savefig.transparent"]='True'
    plt.rcParams['lines.linewidth']=0.8
    plt.rcParams['lines.markersize'] = 4
 
    fig1, ax1 = plt.subplots(nrows=fig_subp[0], ncols=fig_subp[1], \
    sharex=True, figsize=(fig_wh[0], fig_wh[1]) )
    
    return fig1, ax1




def phy_const(sym):
    pc={}
    pc['Bohr2Ang'] = 5.29177210903e-1      
    pc['Ry2eV'] = 13.605693112994   
    pc['qe'] = 1.602176634e-19 

    y = pc[sym]
    return y



def vasp_read_post_data(filename='y_post_data'):
    jobn = []
    Etot = np.array([])
    Eent = np.array([])
    pres = np.empty( [0, 6] )


    f = open(filename, 'r')
    next(f)
    for line in f:
        # to skip unfinished job
        if ( np.abs( float(line.split( )[1]) ) > 1e-6 ) \
        and ( len( line.split() ) > 5 ) :  
            jobn.append( line.split()[0]) 
            Etot = np.append( Etot, float(line.split()[1]) )
            Eent = np.append( Eent, float(line.split()[2]) )

            temp = np.array([])
            for i in np.arange(6):
                temp = np.append(temp, float(line.split()[5+i]) )
            pres = np.vstack( (pres, temp) ) 
    f.close()

    print('==> CHECK data:')
    print(jobn, Etot, Eent, pres)

    return jobn, Etot, Eent, pres




def read_pressure(filename='OUTCAR'):
    y=0
    for line in open(filename):
        if 'in kB' in line: 
            del y
            y=line
  
    if y == 0:
        import sys
        sys.exit("\n==> ABORT: no pressure found in OUTCAR \n" )

    # the last line
    pres = np.array([])
    for i in np.arange(6):
        pres = np.append(pres, float(y.split()[2+i]) )
    print(pres)
    return pres




    
