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
    tmp.create_random_alloys(*args, **kwargs)
    

def create_twin(*args, **kwargs):
    from myvasp import vasp_create as tmp 
    tmp.create_twin(*args, **kwargs)
    




# vasp_io.py


def bestsqs_to_POSCAR(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    tmp.bestsqs_to_POSCAR(*args, **kwargs)



def write_poscar(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    tmp.write_poscar(*args, **kwargs)







def get_list_of_outcar(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms = tmp.get_list_of_outcar(*args, **kwargs)
    return atoms


def get_list_of_atoms(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms = tmp.get_list_of_atoms(*args, **kwargs)
    return atoms


def get_list_of_atoms_from_poscars2(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    latoms = tmp.get_list_of_atoms_from_poscars2(*args, **kwargs)
    return latoms




def my_read_vasp(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms = tmp.my_read_vasp(*args, **kwargs)
    return atoms


def my_write_vasp(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    tmp.my_write_vasp(*args, **kwargs)




def my_read_doscar(*args, **kwargs):
    from myvasp import vasp_io as tmp 
    atoms_dos = tmp.my_read_doscar(*args, **kwargs)
    return atoms_dos 





def my_rm(filename):
    try:
        os.remove(filename)
    except OSError:
        pass





def my_rm_dir(dirname):
    try:
        delete_folder(dirname) 
    except:
        print('Folder does not exist.')


def delete_folder(dirname):
    for i in os.listdir(dirname):
        path_file = os.path.join(dirname, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            delete_folder(path_file)
    os.removedirs(dirname)




def my_read_line(f):
    return np.array([float(x) for x in f.readline().split()])






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


def confirm_int(x, str1='hint'):
    temp = np.linalg.norm( x - np.around(x) )
    if temp > 1e-10:
        print(temp)
        str2 = 'ABORT: x is not int - %s'  %(str1) 
        sys.exit(str2)
    

def confirm_0(x, str1='hint'):
    temp = np.linalg.norm( x )
    if temp > 1e-10:
        print(temp)
        str2 = 'ABORT: x is not 0 - %s'  %(str1) 
        sys.exit(str2) 
    



def calc_RMSE(a, b):
    RMSE = np.sqrt( np.mean( (a-b)**2 ) )
    return RMSE 





def my_interp(x, y, x_new):
    from scipy import interpolate
    func1 = interpolate.interp1d(x, y)
    return func1(x_new)






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





def my_text(ax1, str1, x, y, ha='left', weight='normal'):
    fig_xlim = ax1.get_xlim()
    fig_ylim = ax1.get_ylim()

    ax1.text( 
        fig_xlim[0] + np.diff(fig_xlim) *x, \
        fig_ylim[0] + np.diff(fig_ylim) *y, str1, \
        horizontalalignment=ha, weight=weight )






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

    # print('==> CHECK data:')
    # print(jobn, Etot, Eent, pres)

    return jobn, Etot, Eent, pres






def vasp_read_dir(dirname):
    os.chdir(dirname)
    print( os.getcwd() )
    jobn, Etot, Eent, pres = vasp_read_post_data()
    latoms = get_list_of_atoms()
    os.chdir('..')
    return jobn, Etot, Eent, pres, latoms





def calc_s_from_pres(pres1):
    pres1 = pres1*(-0.1)
    njobs = pres1.shape[0]
    s1 = np.zeros([ njobs, 3, 3 ])
    
    for i in np.arange(njobs):
        for j in np.arange(3):
            s1[i, j, j] = pres1[i, j]
       
        s1[i, 0, 1] = pres1[i, 3]
        s1[i, 0, 2] = pres1[i, 5]
        s1[i, 1, 2] = pres1[i, 4]
    
        s1[i, 1, 0] = s1[i, 0, 1]
        s1[i, 2, 0] = s1[i, 0, 2]
        s1[i, 2, 1] = s1[i, 1, 2]

    return s1 




def normalize_mm(mm):
    for i in np.arange(3):
        mm[i,:] = mm[i,:] / np.linalg.norm(mm[i,:])

    confirm_0( np.dot( mm[0,:], mm[1,:]) )
    confirm_0( np.dot( mm[0,:], mm[2,:]) )
    confirm_0( np.dot( mm[1,:], mm[2,:]) )

    if np.dot( np.cross(mm[0,:], mm[1,:]), mm[2,:] ) <= 0:
        sys.exit('ABORT: wrong mm order.')

    return mm 



    
def rotate_stress(s1, mm):
    mm = normalize_mm(mm) 
    s1r = mm @ s1 @ mm.T
    return s1r  








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




    












def calc_strain(latt1, latt2):
    # latt1 - ref
    # latt2 - deformed 

    F = np.linalg.solve(latt1, latt2)
    F = F.T 
   
    # ================================
    e_ss      = 1/2*( F+F.T ) - np.eye(3)
    e_green   = 1/2*( F.T@F - np.eye(3) )
    e_almansi = 1/2*( np.eye(3) - np.linalg.inv(F@F.T) )

    e_ss_V = calc_to_Voigt(e_ss)

    # ================================
    filen = 'y_post_calc_strain.txt'
    f = open(filen,"w+")

    f.write('# strain of latt2 with respect to latt1: \n\n' )
    with np.printoptions(linewidth=200, \
        precision=8, suppress=True):

        f.write('latt1 \n')
        f.write(str(latt1)+'\n\n')
  
        f.write('latt2 \n')
        f.write(str(latt2)+'\n\n\n')

        f.write('deformation gradient F \n')
        f.write(str(F)+'\n\n')

        f.write('strain e_ss e_green e_almansi \n')
        f.write(str(e_ss)+'\n\n')
        f.write(str(e_green)+'\n\n')
        f.write(str(e_almansi)+'\n\n')
        
        f.write('strain (Voigt notation) e_ss_V \n')
        f.write(str(e_ss_V)+'\n\n')
    f.close()

    return e_ss_V 


def calc_to_Voigt(e):
    e2 = np.zeros(6)
    e2[0] = e[0, 0]
    e2[1] = e[1, 1]
    e2[2] = e[2, 2]
    e2[3] = e[1, 2] + e[2, 1]
    e2[4] = e[0, 2] + e[2, 0]
    e2[5] = e[0, 1] + e[1, 0]
    return e2 







def crystal_shell(struc):
    if struc == 'fcc':
        print('==> fcc ')  

        rcrys = np.sqrt( np.array([ 
        0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ,  5.5,
        6. ,  6.5,  7.5,  8. ,  8.5,  9. ,  9.5, 10. , 10.5, 11. , 11.5,
       12. , 12.5, 13. , 13.5, 14. , 14.5, 15.5, 16. , 16.5, 17. , 17.5,
       18. , 18.5, 19. , 19.5, 20. , 20.5, 21. , 21.5, 22. , 22.5, 23.5,
       24. , 24.5, 25. , 25.5, 26. , 26.5, 27. , 27.5, 28.5, 29. , 29.5,
       30. , 30.5, 31.5, 32. , 32.5, 33. , 33.5, 34. , 34.5, 35. , 35.5,
       36. , 36.5, 37. , 37.5, 38. , 38.5, 39.5, 40. , 40.5, 41. , 41.5,
       42. , 42.5, 43. , 43.5, 44. , 44.5, 45. , 45.5, 46. , 46.5, 47.5,
       48. , 48.5, 49. , 49.5, 50. , 50.5, 51. , 51.5, 52. , 52.5, 53. ,
       53.5, 54. , 54.5, 55.5, 56. , 56.5, 57. , 57.5, 58. , 58.5, 59. ,
       59.5, 60.5, 61. , 61.5, 62. , 62.5, 63.5, 64. , 64.5, 65. , 65.5,
       66. , 66.5, 67. , 67.5, 68. , 68.5, 69. , 69.5, 70. , 70.5, 71.5,
       72. , 72.5, 73. , 73.5, 74. , 74.5, 75. , 75.5, 76. , 76.5, 77. ,
       77.5, 78. , 78.5, 79.5, 80. , 80.5, 81. ])
       )
         
        ncrys = np.array([ 
        12.,   6.,  24.,  12.,  24.,   8.,  48.,   6.,  36.,  24.,  24.,
        24.,  72.,  48.,  12.,  48.,  30.,  72.,  24.,  48.,  24.,  48.,
         8.,  84.,  24.,  96.,  48.,  24.,  96.,   6.,  96.,  48.,  48.,
        36., 120.,  24.,  48.,  24.,  48.,  48., 120.,  24., 120.,  96.,
        24., 108.,  30.,  48.,  72.,  72.,  32., 144.,  96.,  72.,  72.,
        48., 120., 144.,  12.,  48.,  48., 168.,  48.,  96.,  48.,  48.,
        30., 192.,  24., 120.,  72.,  96.,  96.,  24., 108.,  96., 120.,
        48., 144.,  24., 144.,  24.,  96.,  72., 144.,  48., 144.,  48.,
         8., 240.,  54., 120.,  84.,  72.,  48., 240.,  24.,  96.,  72.,
        72.,  96., 120., 144.,  48.,  96.,  48., 240.,  24., 216.,  72.,
        96., 132.,  72., 144.,  96., 144., 192.,   6.,  96.,  96.,  72.,
        96., 240.,  24., 192.,  48., 144.,  96., 168.,  48.,  96., 144.,
        36., 240.,  48., 168., 120.,  72.,  56., 144.,  24., 240.,  96.,
        96.,  48., 312., 144.,  24.,  96., 102.])
           

    elif struc == 'hcp':
        print('==> hcp ')
        
        rcrys = np.array([
            0.707, 0.999, 1.154, 1.224, 1.354, \
            1.414, 1.581, 1.683, 1.732, 1.779, \
            1.825, 1.870, 1.914,  ])
        
        ncrys = np.array([
            12,  6,  2, 18, 12,  \
             6, 12, 12,  6,  6,  \
            12, 24,  6,  ])


    else:
        sys.exit('ABORT: no data in crystal shell. ')
    return rcrys, ncrys











def split_train_test(x, ntrain):
    ntest = x.shape[0] - ntrain     
    if ntest < 0.9:
        x1 = x.copy()
        x2 = []        
    else:        
        temp = int(-1*ntest)
        x1 = x[0: temp].copy()
        x2 = x[temp: ].copy()
        if len( x.shape ) == 1:
            confirm_0( np.hstack([x1, x2]) - x )
        else:
            confirm_0( np.vstack([x1, x2]) - x )    
    return x1, x2




def my_save_pkl(a, filename):
    import pickle
    with open(filename, 'wb') as file:
        pickle.dump(a, file)



def my_read_pkl(filename):
    import pickle
    with open(filename, 'rb') as file:
        a = pickle.load(file)
    return a 





