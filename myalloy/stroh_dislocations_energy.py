
# schematic:
# https://ars.els-cdn.com/content/image/1-s2.0-S1359645416305808-fx2_lrg.jpg 
  


import numpy as np
import scipy as sc
from scipy import integrate
import sys 



def calc_Ec(stroh_u1s1, stroh_u2s2, r0, R0, X1, Y1, X2, Y2, cut1, cut2):
  
# Ec, 4*5:
#        C1   R   C2   1   2
# s1u1
# s2u1
# s1u2
# s2u2


    # energy contributions, integrand
    def fec_u1(x, y, n0):
        u1 = stroh_u1s1(x=x, y=y)[0] 
        s1 = stroh_u1s1(x=x, y=y)[1] 
        s2 = stroh_u2s2(x=x, y=y)[1] 

        ec1 = ((s1 @ n0).T @ u1 ) / 2.0
        ec2 = ((s2 @ n0).T @ u1 ) / 2.0
        
        return ec1, ec2


    def fec_u2(x, y, n0):
        u2 = stroh_u2s2(x=x, y=y)[0] 
        s1 = stroh_u1s1(x=x, y=y)[1] 
        s2 = stroh_u2s2(x=x, y=y)[1] 

        ec3 = ((s1 @ n0).T @ u2 ) / 2.0
        ec4 = ((s2 @ n0).T @ u2 ) / 2.0
        
        return ec3, ec4



    Ec = np.zeros([4,5])



    # S(C1) =====================================
    print('==> S_C1')
    def S_C1(alpha):
        x = X1+r0*np.cos(alpha)
        y = Y1+r0*np.sin(alpha)
        n0 = np.array([[-np.cos(alpha)], [-np.sin(alpha)], [0.]])
        return x, y, n0

    def fec_u1_C1(alpha):
        x, y, n0 = S_C1(alpha)
        return fec_u1(x, y, n0)  

    def fec_u2_C1(alpha):
        x, y, n0 = S_C1(alpha)
        return fec_u2(x, y, n0)  
   
    Ec[0:2,0] = myint(fec_u1_C1, cut1, cut1+2*np.pi)*(r0)
    Ec[2:4,0] = myint(fec_u2_C1, cut2, cut2+2*np.pi)*(r0)


 
    # S(R) ======================================
    print('==> S_R')
    def S_R(alpha):
        x = R0*np.cos(alpha)
        y = R0*np.sin(alpha)
        n0 = np.array([[np.cos(alpha)], [np.sin(alpha)], [0.]])
        return x, y, n0

    def fec_u1_R(alpha):
        x, y, n0 = S_R(alpha)
        return fec_u1(x, y, n0)
    
    def fec_u2_R(alpha):
        x, y, n0 = S_R(alpha)
        return fec_u2(x, y, n0)
    
    Ec[0:2,1]  = myint(fec_u1_R, cut1, cut1+2*np.pi)*(R0) 
    Ec[2:4,1]  = myint(fec_u2_R, cut2, cut2+2*np.pi)*(R0) 



    # S(C2) =====================================
    print('==> S_C2')
    def S_C2(alpha):
        x = X2+r0*np.cos(alpha)
        y = Y2+r0*np.sin(alpha)
        n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
        return x, y, n0

    def fec_u1_C2(alpha):
        x, y, n0 = S_C2(alpha)
        return fec_u1(x, y, n0)
      
    def fec_u2_C2(alpha):
        x, y, n0 = S_C2(alpha)
        return fec_u2(x, y, n0)

    Ec[0:2,2]  = myint(fec_u1_C2, cut1, cut1+2*np.pi)*(r0) 
    Ec[2:4,2]  = myint(fec_u2_C2, cut2, cut2+2*np.pi)*(r0) 



    # S(1) ======================================
    print('==> S_1')
    def S_11():    # S(1+)
        y = Y1+1e-100
        n0 = [[0.],[-1.],[0.]]
        return y, n0

    def fec_u1_11(x):
        y, n0 = S_11() 
        return fec_u1(x, y, n0)

    def S_12():   # S(1-)
        y = Y1-1e-100
        n0 = [[0.],[1.],[0.]]
        return y, n0

    def fec_u1_12(x):
        y, n0 = S_12() 
        return fec_u1(x, y, n0)

    Ec[0:2,3] = myint(fec_u1_11, -sc.sqrt(R0**2-Y1**2), X1-r0 ) \
              + myint(fec_u1_12, -sc.sqrt(R0**2-Y1**2), X1-r0 ) 



    # S(2) ======================================
    print('==> S_2')
    def S_21():    # S(2+)
        y = Y2+1e-100
        n0 = [[0.],[-1.],[0.]]
        return y, n0

    def fec_u2_21(x):
        y, n0 = S_21() 
        return fec_u2(x, y, n0)

    def S_22():   # S(2-)
        y = Y2-1e-100
        n0 = [[0.],[1.],[0.]]
        return y, n0

    def fec_u2_22(x):
        y, n0 = S_22() 
        return fec_u2(x, y, n0)

    Ec[2:4,4] = myint(fec_u2_21, X2+r0, sc.sqrt(R0**2-Y2**2)  ) \
              + myint(fec_u2_22, X2+r0, sc.sqrt(R0**2-Y2**2)  ) 




    print('==> Ec:')
    print(Ec)
    check_Ec(Ec)
    return Ec 










# integrate each return of f(x)
def myint(f, x1, x2):
    nf = len(f(1))   # number of returns of f(x)

    F = np.array([])
    for i in np.arange(nf):  
        def g(x):
            return f(x)[i]

        temp = integrate.quad(g, x1, x2 , epsabs=0, limit=100)
        F = np.append(F, temp[0])

    print('==> myint done.')
    return F








def check_Ec(Ec):
    tola = 1.0e-6

    temp = np.abs(Ec[0,0]/Ec[0,1]+1)
    if temp > tola:
        print(temp)
        sys.exit('ABORT: wrong Ec[0,0] in Ec.')

    temp = np.abs(Ec[3,1]/Ec[3,2]+1) *1e-2
    if temp > tola:
        print(temp)
        sys.exit('ABORT: wrong Ec[3,1] in Ec.')

    temp = np.abs( np.sum(Ec[1,:]) / np.sum(Ec[2,:]) -1 )
    if temp > tola:
        print(temp)
        sys.exit('ABORT: wrong Ec[1,:] in Ec.')

   



