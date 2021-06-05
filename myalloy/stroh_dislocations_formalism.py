

import numpy as np 
import sys 


def calc_N_p_A_B(c):

    Q = np.zeros([3, 3])
    R = np.zeros([3, 3])
    T = np.zeros([3, 3])
    for i in np.arange(0,3,1):
        for k in np.arange(0,3,1):
            Q[i, k] = c[i, 0, k, 0]
            R[i, k] = c[i, 0, k, 1]
            T[i, k] = c[i, 1, k, 1]

    Ti = np.linalg.inv(T)

    N11 = -Ti @ (R.T)
    N21 = R @ Ti @ (R.T) -Q
    N22 = -R @ Ti

    N = np.zeros([6,6])
    for i in np.arange(0,3,1):
        for j in np.arange(0,3,1):
            N[i,j]     = N11[i,j]
            N[i,j+3]   = Ti[i,j]
            N[i+3,j]   = N21[i,j]
            N[i+3,j+3] = N22[i,j]

    va1, ve = np.linalg.eig(N)     #va-eigenvalue;  ve-eigenvector
    va = np.zeros([6,6], dtype = complex)
    for i in np.arange(0,6,1):
        va[i,i] = va1[i]

    k1 = -1
    k2 = 2
    p  = np.zeros([6,1], dtype = complex )  
    xi = np.zeros([6,6], dtype = complex )   


    for i in np.arange(0,6,1):
        if np.imag(va[i,i]) > 0:
            k1 = k1+1
            p[k1,0] = va[i,i]
            xi[:,k1] = ve[:,i]
        else:
            k2 = k2+1
            p[k2,0] = va[i,i]
            xi[:,k2] = ve[:,i]

    for i in np.arange(0,3,1):
        t1 = abs( p[i,0].real / p[i+3,0].real -1 )
        t2 = abs( p[i,0].imag / p[i+3,0].imag -1 ) 
        if t1 > 1e-6 or t2 > 1e-6 :     
            print(p, i, t1, t2)  
            sys.exit('ABORT: wrong order of p') 

    J = np.zeros([6,6])
    for i in np.arange(3):
        J[i,i+3] = 1
        J[i+3,i] = 1

    nxi = np.zeros([6,6],dtype = complex)
    for i in np.arange(6):
        temp = xi[:,i]
        nxi[:,i] = temp/np.sqrt(temp.T @ J @ temp)

    
    A = np.zeros([3,3],dtype = complex)
    B = np.zeros([3,3],dtype = complex)
    for i in np.arange(0,3,1):
        A[:,i] = nxi[0:3, i]
        B[:,i] = nxi[3:6, i]
        
    return N, p, A, B






def calc_K(b1, b2, B):
    K1  = 1.0/(2.0*np.pi) * b1.T @ np.imag(B @ B.T) @ b1
    K2  = 1.0/(2.0*np.pi) * b2.T @ np.imag(B @ B.T) @ b2
    K12 = 1.0/np.pi * b2.T @ np.imag(B @ B.T) @ b1
        
    print('==> K1, K2, K12 [eV/Ang]:')
    print( K1, K2, K12 )
    return K1, K2, K12








def calc_Er_Et(theta, b1, b2, r0,   p, B, K12, r12):
    Er = K12*np.log(r0/r12)

    ft = np.zeros([3,3], dtype = complex)
    for i in np.arange(3):
        ft[i,i] = np.log( np.cos(theta) + p[i,0] * np.sin(theta) )
    Et = 1/np.pi * b2.T @ np.imag(B @ ft @ B.T) @ b1
    
    return Er, Et 












# displacement u0 and stress s0

def myacos(y, x):
    r = np.sqrt(x**2 + y**2)
    hy = (np.heaviside(y,0.5)-0.5) *2.0 \
        + np.heaviside(y,0.5) *np.heaviside(-y,0.5) *4.0
    t = (1.0-hy)*np.pi + hy*np.arccos(x/r)
    return t





def stroh_u0_s0(p, A, B, b, X, Y, cut,  x, y):

    fz     = np.zeros([3,3], dtype = complex)
    P      = np.zeros([3,3], dtype = complex)
    Lambda = np.zeros([3,3], dtype = complex)

    for i in np.arange(3):
        tx = (x-X)+np.real(p[i,0])*(y-Y)
        ty =       np.imag(p[i,0])*(y-Y)
        tr = np.sqrt(tx**2+ty**2)
        
        if cut == 0:
            fz[i,i] = np.log(tr) + 1.0j * myacos(ty, tx)
        elif cut == -np.pi:
            fz[i,i] = np.log(tr) + 1.0j * np.arctan2(ty, tx)   

        P[i,i] = p[i,0]
        Lambda[i,i] = 1.0/((x-X)+p[i,0]*(y-Y))
    
    u0 = np.imag(A @ fz @ B.T) @ b /np.pi

    S1 = -1.0/np.pi* np.imag( B @ P @ Lambda @ B.T) @ b       
    S2 =  1.0/np.pi* np.imag( B @ Lambda @ B.T ) @ b       

    s0 = np.zeros([3,3])
    
    s0[0:3,0] = S1[0:3,0]
    s0[0:3,1] = S2[0:3,0]
   
    s0[0,2] = s0[2,0]
    s0[1,2] = s0[2,1]
    
    return u0, s0
    



