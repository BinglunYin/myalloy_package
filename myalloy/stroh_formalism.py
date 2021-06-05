
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
    
    print(Ec)





#     r12=8
#     LMdata='T'
#     Ldispl='T'

#  #   qe, R0, r0, c, C, c, N, p, A, B, K1, K2, K12, X1, Y1, X2, Y2, Ec, gamma, r12, Ect1, Ect2, Ect3, Er, Etotm 
#     K12, r12, disptot = stroh_aniso_energy(self, mm, b1, b2, theta, r12, LMdata, Ldispl)
#     print(K12, r12)
#     print(disptot)





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
        
    print('==>  K1, K2, K12 [eV/Ang]:', K1, K2, K12 )
    return K1, K2, K12







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
        









    
    
    
#     Ect1 = np.zeros([1,5])
#     for i in np.arange(0,5,1):
#         Ect1[0,i] = sum(Ec.T[i])
#     #print(Ect1)

#     Ect2 = np.zeros([4,1])
#     for i in np.arange(0,4,1):
#         Ect2[i,0] = sum(Ec[i])
#     #print(Ect2)

#     Ect3 = sum(Ect1[0])
#     Esf = gamma*r12
#     Etot = Ect3+Esf
    
# #  stroh_check_Ec   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     tola = 1.0e-6
#     check_value = 0
#     if abs(Ec[0,0]/Ec[0,1]+1) > tola:
#         check_value = check_value + 1
#         print('wrong Ec(1,1) in A')
#     elif abs(Ec[3,1]/Ec[3,2]+1) > tola*1.0e+2:
#         check_value = check_value + 1
#         print('wrong Ec(4,2) in A')

#     if abs( sum(Ec[1]) / sum(Ec[2]) -1 ) > tola:
#         check_value = check_value + 1
#         print('wrong D in Ec')

#     if check_value == 0:
#         print('****************************************************')
#         print('***********   Ec all checked!'' *******************')
#         print('****************************************************')

#     Er = K12*np.log(r0/r12)
#     ft = np.zeros([3,3], dtype = complex)
#     for i in arange(0,3,1):
#         ft[i,i] = np.log( np.cos(theta) + p[i,0] * np.sin(theta) )
#     Et = 1/np.pi * b2.T @ np.imag(B @ ft @ B.T) @ b1
    
#     Etotm = Er-Et+Esf

#     np.set_printoptions(linewidth=100)
#     if LMdata == 'T':
# #	pre_cwd = os.getcwd()
# #	path = pre_cwd    
#         Pdata_name = 'Pdata_' + brav_latt +'_elem_'+ '_r12a_' + "%4.1f"%(r12/a)
#         with open(Pdata_name,'w') as w:
#             w.write('     a        c/a        r0/a        R0/a  \n')
#             w.write("%12.8f"%(a)+"%12.8f"%(c/a)+'%12.8f'%(r0/a)+"%15.8e"%((R0/a))+'\n\n')
            
#             w.write('    theta(rad)   theta(degree)   gamma(eV/Ang^2)   gamma(mJ/m^2)   \n')
#             w.write("%15.8f"%(theta) + "%15.8f"%(theta/np.pi*180) + "%18.8f"%(gamma) + "%15.8f"%(gamma*(qe*1000)/(1e-20))+'\n\n')
            
#             w.write('mm\n')
#             w.write(str(mm)+'\n\n')
            
#             w.write('         b1             b2            b1+b2 \n')
#             for i in arange(0,len(b1)):
#                 w.write("%15.8f"%(b1[i])+"%15.8f"%(b2[i])+"%15.8f"%((b1+b2)[i])+'\n')
#             w.write('\n')
            
#             w.write('norm(b1,2)   norm(b2,2)   norm(b1+b2,2)\n')
#             w.write(str(np.linalg.norm(b1))+'   '+str(np.linalg.norm(b2))+'   '+str(np.linalg.norm(b1+b2))+'   \n\n')
            
#             w.write('Cij(GPa) before rotation\n')
#             w.write(str(c)+'\n\n')
            
#             w.write('Cij(GPa) after rotation\n')
#             w.write(str(C)+'\n\n')
            
#             w.write('N \n')
#             w.writelines(str(N)+'\n\n')
            
#             w.write('p \n')
#             w.write(str(p)+'\n\n')
            
#             w.write('A \n')
#             w.write(str(A)+'\n\n')        

#             w.write('B \n')
#             w.write(str(B)+'\n\n')
            
#             w.write('     K1           K2         K12 \n')
#             w.write("%12.8f"%(K1[0])+"%12.8f"%(K2[0])+"%12.8f"%(K12[0])+'\n\n')
            
#             w.write('     r12        r12/a       r12/b      r12/r0 \n')
#             w.write("%12.8f"%(r12)+"%12.8f"%(r12/a)+"%12.8f"%(r12/np.linalg.norm(b1+b2))+"%12.8f"%(r12/r0)+'\n\n')
            
#             w.write('     X1          Y1         X2          Y2 \n')
#             w.write("%12.8f"%(X1)+"%12.8f"%(Y1)+"%12.8f"%(X2)+"%12.8f"%(Y2)+'\n\n')
            
#             w.write('Energy contribution, Ec (eV/Ang) \n')
#             w.write(str(Ec)+'\n\n')
            
#             w.write(' Ect1 \n')
#             w.write(str(Ect1)+'\n\n')
            
#             w.write(' Ect2 \n')
#             w.write(str(Ect2)+'\n\n')
            
#             w.write('    Ect3        Esf         Etot \n')
#             w.write("%12.8f"%(Ect3)+"%12.8f"%(Esf)+"%12.8f"%(Etot)+'\n\n')
            
#             w.write('      Er          Et          Esf        Etotm \n')
#             w.write("%12.8f"%(Er)+"%12.8f"%(Et)+"%12.8f"%(Esf)+"%12.8f"%(Etotm)+'\n\n')
            
#             w.close
#         os.chdir('../')

#     return K12, r12, disptot
























