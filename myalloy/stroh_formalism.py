
# By T. Liu and B. Yin
# 2021.06.04



import numpy as np
import sys 



def calc_stroh(self, slip_system='basal_a_edge', bp=None, param={}):
    qe=1.60217657e-19

    if not hasattr(self, 'c'):
        self.c = self.a

    from myalloy import stroh_formalism_slip_system as ss
    mm, bt, theta, xx, yy, b1, b2 = \
        ss.stroh_slip_system(self, slip_system=slip_system, bp=bp)
    
    a   = self.a
    c   = self.c
    Cij = self.Cij 
    brav_latt = self.brav_latt
    
    gamma = self.gamma * 1e-3 * (1 / qe / 1e20)  # to [eV/Ang^2]

    r0=0.5*np.linalg.norm(b1+b2)
    R0 = 1e7*a
    
   

    from myalloy import calc_elastic_constant as cec 
    E2, CIJ2 = cec.rotate_Cij(brav_latt, Cij, mm)

    E2 = E2*1.0e9*(1.0/qe/1.0e30)     # to [eV/Ang^3]

    N, p, A, B = calc_N_p_A_B(E2)
    K1, K2, K12 = calc_K(b1, b2, B) 
    

    if 'r12' in param: 
        r12 = param['r12']
    else:
        r12 = K12/gamma

    if r12 < 2.0*r0 :
        print(r12, 2.0*r0)
        sys.exit('ABORT: r12<2*r0.')


    Er, Et = calc_Er_Et(K12, r0, r12, theta, p, B, b1, b2)




    X1 = 0.
    Y1 = 0.
    X2 = X1+r12*np.cos(theta)
    Y2 = Y1+r12*np.sin(theta)
    cut1 = -np.pi
    cut2 = 0 
    

    from functools import partial
    stroh_u1s1 = partial(stroh_u0_s0, p=p, A=A, B=B, \
        b=b1, X=X1, Y=Y1, cut=cut1)
    
    stroh_u2s2 = partial(stroh_u0_s0, p=p, A=A, B=B, \
        b=b2, X=X2, Y=Y2, cut=cut2)

    
    if 'pos_in' in param: 
        calc_pos_out(stroh_u1s1, stroh_u2s2, param['pos_in'])




    from myalloy import stroh_formalism_energy 
    Ec = stroh_formalism_energy.calc_Ec(stroh_u1s1, stroh_u2s2, \
         r0, R0, X1, Y1, X2, Y2)
    
    print('==> Ec:')
    print(Ec)


    if 'output_name' in param: 
        write_output(param['output_name'], qe, \
            a, c, r0, R0, theta, b1, b2, gamma, \
            K1, K2, K12, r12, X1, Y1, X2, Y2, Er, Et, Ec, \
            mm, CIJ2, CIJ2, N, p, A, B )

    













# ===========================

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
        t11 =        abs(np.real(p[i,0] - p[i+3,0]) )
        t12 = 1.0e-6*abs(np.real(p[i,0]) ) 
        t21 =        abs(np.imag(p[i,0] + p[i+3,0]) )
        t22 = 1.0e-6*abs(np.imag(p[i,0]) )
        if t11 > t12 or t21 > t22:     
            print(p, i, t11, t12, t21, t22)  
            sys.exit('ABORT: wrong order of p') 

    J = np.zeros([6,6])
    for i in np.arange(0,3,1):
        J[i,i+3] = 1
        J[i+3,i] = 1

    nxi = np.zeros([6,6],dtype = complex)
    for i in np.arange(0,6,1):
        tnx1 = xi[:,i]
        nxi[:,i] = tnx1/np.sqrt(tnx1.T @ J @ tnx1)

    
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






def calc_Er_Et(K12, r0, r12, theta, p, B, b1, b2):
    Er = K12*np.log(r0/r12)

    ft = np.zeros([3,3], dtype = complex)
    for i in np.arange(3):
        ft[i,i] = np.log( np.cos(theta) + p[i,0] * np.sin(theta) )
    Et = 1/np.pi * b2.T @ np.imag(B @ ft @ B.T) @ b1
    
    return Er, Et 






# ===========================
   
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
    S2 = 1.0/np.pi* np.imag( B @ Lambda @ B.T ) @ b       

    s0 = np.zeros([3,3])
    
    s0[0:3,0] = S1[0:3,0]
    s0[0:3,1] = S2[0:3,0]
   
    s0[0,2] = s0[2,0]
    s0[1,2] = s0[2,1]
    
    return u0, s0
    







def calc_pos_out(stroh_u1s1, stroh_u2s2, pos_in):
      
    natoms = pos_in.shape[0]
    disp1 = np.empty(shape=(natoms,3))
    disp2 = np.empty(shape=(natoms,3))
        
    for i in np.arange(natoms):
        disp1[i,:] = stroh_u1s1(x=pos_in[i,0], y=pos_in[i,1])[0].T
        disp2[i,:] = stroh_u2s2(x=pos_in[i,0], y=pos_in[i,1])[0].T
    
    pos_out = pos_in + disp1 + disp2
    np.savetxt('stroh_pos_out.txt', pos_out)
        







def write_output(output_name, qe, \
    a, c, r0, R0, theta, b1, b2, gamma, \
    K1, K2, K12, r12, X1, Y1, X2, Y2, Er, Et, Ec, \
    mm, CIJ, CIJ2, N, p, A, B ):

    np.set_printoptions(linewidth=1000)


    filen = 'stroh_' + output_name + '.txt'
    f = open(filen,"w+")


    f.write('# stroh formalism for two dislocations: \n' )
    
    f.write('%16s %16s %16s %16s \n' \
        %('a', 'c/a', 'r0/a', 'R0/a' ) )
    f.write('%16.8f %16.8f %16.8f %16.1e \n\n' \
        %(a, c/a, r0/a, R0/a) )

   
    f.write('%16s %16s %16s %16s \n' \
        %('theta(rad)', 'theta(degree)', 'gamma(eV/Ang^2)', 'gamma(mJ/m^2)' ) )
    f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
        %(theta, theta/np.pi*180, gamma, gamma*(qe*1000)/(1e-20)) )

  
            
    f.write('%16s %16s %16s \n' \
        %('b1', 'b2', 'b1+b2' ) )
    for i in np.arange(3):
        f.write('%16.8f %16.8f %16.8f \n' \
        %(b1[i], b2[i], b1[i]+b2[i]) )            
    f.write('\n')
    

    f.write('%16s %16s %16s \n' \
        %('K1', 'K2', 'K12' ) )
    f.write('%16.8f %16.8f %16.8f \n\n' \
        %(K1, K2, K12) )


    f.write('%16s %16s %16s %16s \n' \
        %('r12 (Ang)', 'r12/a', 'r12/b', 'r12/r0' ) )
    f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
        %(r12, r12/a, r12/np.linalg.norm(b1+b2), r12/r0) )

    f.write('%16s %16s %16s %16s \n' \
        %('X1', 'Y1', 'X2', 'Y2' ) )
    f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
        %(X1, Y1, X2, Y2) )



    Esf = gamma*r12
    f.write('%16s %16s %16s \n' \
        %('E_r', 'E_t', 'E_sf' ) )
    f.write('%16.8f %16.8f %16.8f \n\n' \
        %(Er, Et, Esf) )


    f.write('%16s %16s \n' \
        %('Ec.sum()', 'E_tot' ) )
    f.write('%16.8f %16.8f \n\n' \
        %(Ec.sum(), Ec.sum()+Esf ) )




    f.write('Energy contribution, Ec (eV/Ang) \n')
    f.write(str(Ec)+'\n\n')
            
       







    f.write('mm\n')
    f.write(str(mm)+'\n\n')
       
    f.write('Cij(GPa) before rotation\n')
    f.write(str(CIJ)+'\n\n')
            
    f.write('Cij(GPa) after rotation\n')
    f.write(str(CIJ2)+'\n\n')
            
    f.write('N \n')
    f.write(str(N)+'\n\n')
            
    f.write('p \n')
    f.write(str(p)+'\n\n')
            
    f.write('A \n')
    f.write(str(A)+'\n\n')        

    f.write('B \n')
    f.write(str(B)+'\n\n')


   
    f.close()



   
         
        
























