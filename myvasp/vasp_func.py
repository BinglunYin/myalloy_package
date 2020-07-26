#!/home/yin/opt/bin/python3

import numpy as np
import os, sys, copy




# vasp_create.py

def create_supercell(*args, **kwargs):
    from myvasp import vasp_create as tmp 
    atoms = tmp.create_supercell(*args, **kwargs)
    return atoms


def make_SFP_xy(*args, **kwargs):
    from myvasp import vasp_create as tmp 
    atoms = tmp.make_SFP_xy(*args, **kwargs)
    return atoms


def make_a3_ortho(*args, **kwargs):
    from myvasp import vasp_create as tmp 
    atoms = tmp.make_a3_ortho(*args, **kwargs)
    return atoms


def create_random_alloys(*args, **kwargs):
    from myvasp import vasp_create as tmp 
    tmp.create_supercell(*args, **kwargs)
    




# vasp_io.py
def get_list_of_outcar(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms = tmp.get_list_of_outcar(*args, **kwargs)
    return atoms


def get_list_of_atoms(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms = tmp.get_list_of_atoms(*args, **kwargs)
    return atoms


def my_read_vasp(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms = tmp.my_read_vasp(*args, **kwargs)
    return atoms


def my_write_vasp(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    tmp.my_write_vasp(*args, **kwargs)


def my_rm(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    tmp.my_rm(*args, **kwargs)








# math

def mylinreg(X, y):
    if X.shape[0] < X.shape[1]:
        sys.exit('==> ABORT. too few data. ')

    temp = np.linalg.matrix_rank(X)
    if temp < X.shape[1]:
        sys.exit('==> ABORT. linear regression ill-conditioned. ')

    beta = np.linalg.inv(X.T @ X) @ X.T @ y 
    SStot = np.sum( (y-y.mean())**2 )
    SSres = np.sum( (y-X@beta)**2 )
    R2 = 1- SSres/SStot
    print('==> beta, R2:', beta, R2)
    return beta, R2




# plot

def my_plot(fig_wh, fig_subp, fig_sharex=True):
    import matplotlib.pyplot as plt

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
    sharex=fig_sharex, figsize=(fig_wh[0], fig_wh[1]) )
    
    return fig1, ax1





# basic func

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




    
