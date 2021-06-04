

import numpy as np
import sys 





def calc_nu_from_B_mu(B, mu):
    nu = (3*B -2*mu)/(3*B +mu)/2
    return nu


    
def calc_nu_from_E_mu(E, mu):
    nu = E/(2*mu) -1
    return nu



def calc_B_from_mu_nu(mu, nu):
    B = 2*mu*(1+nu)/(1-2*nu)/3
    return B



def calc_E_from_mu_nu(mu, nu):
    E = 2*mu*(1+nu)
    return E






def symmetrize_matrix(C_in):
    C = C_in.copy()
    if np.diff(C.shape) != 0:
        sys.exit('ABORT: C has to be a square matrix.') 
    
    for i in np.arange(1, C.shape[0]):
        for j in np.arange(0, i):
            C[i,j] = C[j,i]
   
    return C




# CIJ is 6*6 
def calc_CIJ_from_Cij(brav_latt, Cij):

    if (brav_latt is 'fcc') or (brav_latt is 'bcc'):
        [C11, C12, C44] = Cij
        
        CIJ=np.array([
            [C11, C12, C12,   0,   0,   0],
            [  0, C11, C12,   0,   0,   0],
            [  0,   0, C11,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C44],
        ])
    
    elif (brav_latt is 'hcp'):
        [C11, C12, C13, C33, C44] = Cij
        
        CIJ=np.array([
            [C11, C12, C13,   0,   0,   0],
            [  0, C11, C13,   0,   0,   0],
            [  0,   0, C33,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, (C11-C12)/2],
        ])
    
    CIJ = symmetrize_matrix(CIJ)
    return CIJ






def calc_Cijavg_from_Cij(brav_latt, Cij):
    
    CIJ = calc_CIJ_from_Cij(brav_latt, Cij)
        
    Cijavg={}

    if brav_latt is 'fcc':
        [C11, C12, C44] = Cij
        mu_111 = C44 - ( 2*C44 +C12 -C11 )/3
        Cijavg['mu_111'] = mu_111

    # Zener anisotropy
    Cijavg['A'] = 2*CIJ[3,3]/(CIJ[0,0]-CIJ[0,1])


    # Voigt       
    c1 = CIJ[0,0] +CIJ[1,1] +CIJ[2,2]
    c2 = CIJ[0,1] +CIJ[0,2] +CIJ[1,2]
    c3 = CIJ[3,3] +CIJ[4,4] +CIJ[5,5]
    
    B_V  = (c1 +2*c2)/9
    mu_V = (c1 -c2 +3*c3)/15
    nu_V = calc_nu_from_B_mu(B_V, mu_V)
    
    # Reuss
    SIJ = np.linalg.inv(CIJ)
    s1 = SIJ[0,0] +SIJ[1,1] +SIJ[2,2]
    s2 = SIJ[0,1] +SIJ[0,2] +SIJ[1,2]
    s3 = SIJ[3,3] +SIJ[4,4] +SIJ[5,5]
    B_R  = 1/(s1 +2*s2)
    mu_R = 15/(4*s1 -4*s2 +3*s3)
    nu_R = calc_nu_from_B_mu(B_R, mu_R)
    
    # Hill
    B_H  = (B_V + B_R)/2
    mu_H = (mu_V + mu_R)/2
    nu_H = calc_nu_from_B_mu(B_H, mu_H)
    
    Cijavg.update({ \
        'B_V': B_V, 'mu_V': mu_V, 'nu_V': nu_V, \
        'B_R': B_R, 'mu_R': mu_R, 'nu_R': nu_R, \
        'B_H': B_H, 'mu_H': mu_H, 'nu_H': nu_H, \
    })
 
    return Cijavg








# E is 3*3*3*3, CIJ is 6*6
def calc_E_from_CIJ(CIJ):
    data = np.array([ 
        [1, 1, 1, 1, CIJ[0, 0] ], 
        [2, 2, 2, 2, CIJ[1, 1] ], 
        [3, 3, 3, 3, CIJ[2, 2] ], 

        [1, 1, 2, 2, CIJ[0, 1] ], 
        [2, 2, 1, 1, CIJ[0, 1] ], 

        [1, 1, 3, 3, CIJ[0, 2] ], 
        [3, 3, 1, 1, CIJ[0, 2] ], 

        [2, 2, 3, 3, CIJ[1, 2] ], 
        [3, 3, 2, 2, CIJ[1, 2] ], 

        [2, 3, 2, 3, CIJ[3, 3] ], 
        [2, 3, 3, 2, CIJ[3, 3] ], 
        [3, 2, 2, 3, CIJ[3, 3] ], 
        [3, 2, 3, 2, CIJ[3, 3] ], 
      
        [1, 3, 1, 3, CIJ[4, 4] ], 
        [1, 3, 3, 1, CIJ[4, 4] ], 
        [3, 1, 1, 3, CIJ[4, 4] ], 
        [3, 1, 3, 1, CIJ[4, 4] ], 
      
        [1, 2, 1, 2, CIJ[5, 5] ], 
        [1, 2, 2, 1, CIJ[5, 5] ], 
        [2, 1, 1, 2, CIJ[5, 5] ], 
        [2, 1, 2, 1, CIJ[5, 5] ],  
    ])

    E = np.zeros([3, 3, 3, 3])
    for i in np.arange(data.shape[0]):
        [i, j, k, l] = np.array( data[i, 0:4]-1,  dtype='int' )
        E[i, j, k, l] = data[i, 4]

    return E 





def calc_CIJ_from_E(E):
    CIJ = np.zeros([6,6])
    
    for i in np.arange(1,4,1):
        for j in np.arange(1,4,1):
            for k in np.arange(1,4,1):
                for l in np.arange(1,4,1):
                    if i == j:
                        m = i
                    else:
                        m = 9-i-j
                    
                    if k == l:
                        n = k
                    else:
                        n = 9-k-l
                    
                    CIJ[m-1, n-1] = E[i-1, j-1, k-1, l-1]
    return CIJ 






def rotate_Cij(brav_latt, Cij, mm):

    CIJ = calc_CIJ_from_Cij(brav_latt, Cij)
    E   = calc_E_from_CIJ(CIJ)

    temp = calc_CIJ_from_E(E)
    if np.linalg.norm( CIJ-temp ) > 1e-10:
        print(CIJ, temp, CIJ-temp)
        sys.exit('ABORT: wrong CIJ and E.')
    

    ee = np.eye(3)
    a = np.dot(mm, ee.T)

    # E2 is the rotated E
    E2 = np.zeros([3, 3, 3, 3])
    for i in np.arange(3):
        for j in np.arange(3):
            for k in np.arange(3):
                for l in np.arange(3):

                    for s in np.arange(3):
                        for r in np.arange(3):
                            for q in np.arange(3):
                                for p in np.arange(3):
                                    E2[i,j,k,l] = E2[i,j,k,l] \
                                        + a[i,p]*a[j,q] *E[p,q,r,s] *a[k,r]*a[l,s]

    CIJ2 = calc_CIJ_from_E(E2) 
    return CIJ2





