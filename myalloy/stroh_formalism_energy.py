  
  
import scipy as sc
import numpy as np
from scipy import integrate


# integrate each return of f 
def myint(f, x1, x2):
    F = np.array([])

    for i in len(f):  
        def g(x):
            return f(x)[i]

        temp = integrate.quad(g, x1, x2 )
        F = np.append(F, temp)

    return F







def calc_Ec(stroh_us1, stroh_us2, r0, R0, X1, Y1, cut1, cut2):

# ec11 ~ ec41  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
    def x1y1(alpha):
        x = X1+r0*np.cos(alpha)
        y = Y1+r0*np.sin(alpha)
        n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
        n1 = [[0.],[1.],[0.]]
        return x, y, n0, n1 
    

    def feci1(alpha):
        x, y, n0, n1 = x1y1(alpha) 
        u1 = stroh_us1(x=x, y=y)[0] 
        s1 = stroh_us1(x=x, y=y)[1] 
        u2 = stroh_us2(x=x, y=y)[0] 
        s2 = stroh_us2(x=x, y=y)[1] 

        ec11 = ((s1 @ n0).T @ u1 ) * r0 / 2.0
        ec21 = ((s1 @ n0).T @ u2 ) * r0 / 2.0
        ec31 = ((s2 @ n0).T @ u1 ) * r0 / 2.0
        ec41 = ((s2 @ n0).T @ u2 ) * r0 / 2.0
        
        return ec11, ec21, ec31, ec41
        
    F = myint(feci1, cut1, cut1+2.0*sc.pi)
    eci1 = F.T
    print(eci1)

#     def fec21(alpha):
#         x, y, n0, n1 = x1y1(alpha) 
#         s1 = stroh_us1(x=x, y=y)[1] 
#         u2 = stroh_us2(x=x, y=y)[0]
        
#         return ec
#     ec21 = sc.integrate.quad(fec21, cut2, cut2+2.0*sc.pi )

#     def fec31(alpha):
#         x = X1+r0*np.cos(alpha)
#         y = Y1+r0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
#         u1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[0]
            
        
#         return ec
#     ec31 = sc.integrate.quad(fec31, cut1, cut1+2.0*sc.pi )

#     def fec41(alpha):
#         x = X1+r0*np.cos(alpha)
#         y = Y1+r0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
#         u2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[0]
            
        
#         return ec
#     ec41 = sc.integrate.quad(fec41, cut2, cut2+2.0*sc.pi )

#     #  ec12 ~ ec42 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
#     def fec12(alpha):
#         x = R0*np.cos(alpha)
#         y = R0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[1] 
#         u1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[0]
            
#         ec = -((s1 @ n0).T @ u1 ) * R0 / 2.0
        
#         return ec
#     ec12 = sc.integrate.quad(fec12, cut1, cut1+2.0*sc.pi )

#     def fec21(alpha):
#         x = R0*np.cos(alpha)
#         y = R0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[1] 
#         u2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[0]
            
#         ec = -((s1 @ n0).T @ u2 ) * R0 / 2.0
        
#         return ec
#     ec22 = sc.integrate.quad(fec21, cut2, cut2+2.0*sc.pi )

#     def fec32(alpha):
#         x = R0*np.cos(alpha)
#         y = R0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
#         u1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[0]
            
#         ec = -((s2 @ n0).T @ u1 ) * R0 / 2.0
        
#         return ec
#     ec32 = sc.integrate.quad(fec32, cut1, cut1+2.0*sc.pi )

#     def fec42(alpha):
#         x = R0*np.cos(alpha)
#         y = R0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
#         u2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[0]
            
#         ec = -((s2 @ n0).T @ u2 ) * R0 / 2.0
        
#         return ec
#     ec42 = sc.integrate.quad(fec42, cut2, cut2+2.0*sc.pi )




#     #  ec13 ~ ec43  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
#     def fec13(alpha):
#         x = X2+r0*np.cos(alpha)
#         y = Y2+r0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[1] 
#         u1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[0]
            
#         ec = ((s1 @ n0).T @ u1 ) * r0 / 2.0
        
#         return ec
#     ec13 = sc.integrate.quad(fec13, cut1, cut1+2.0*sc.pi )

#     def fec23(alpha):
#         x = X2+r0*np.cos(alpha)
#         y = Y2+r0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[1] 
#         u2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[0]
            
#         ec = ((s1 @ n0).T @ u2 ) * r0 / 2.0
        
#         return ec
#     ec23 = sc.integrate.quad(fec23, cut2, cut2+2.0*sc.pi )

#     def fec33(alpha):
#         x = X2+r0*np.cos(alpha)
#         y = Y2+r0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
#         u1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[0]
            
#         ec = ((s2 @ n0).T @ u1 ) * r0 / 2.0
        
#         return ec
#     ec33 = sc.integrate.quad(fec33, cut1, cut1+2.0*sc.pi )

#     def fec43(alpha):
#         x = X2+r0*np.cos(alpha)
#         y = Y2+r0*np.sin(alpha)
#         n0 = np.array([[-np.cos(alpha)],[-np.sin(alpha)],[0.]])
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
#         u2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[0]
            
#         ec = ((s2 @ n0).T @ u2 ) * r0 / 2.0
        
#         return ec
#     ec43 = sc.integrate.quad(fec43, cut2, cut2+2.0*sc.pi )



#     #  ec14  ec34  ec25  ec45  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
#     def fec14(x):
#         y = Y1
#         n1 = [[0.],[1.],[0.]]
        
#         s1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[1] 
            
#         ec = (-s1 @ n1).T @ b1 / 2.0
        
#         return ec
#     ec14 = sc.integrate.quad(fec14, -sc.sqrt(R0**2-Y1**2), X1-r0 )

#     def fec34(x):
#         y = Y1
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
            
#         ec = (-s2 @ n1).T @ b1 / 2.0
        
#         return ec
#     ec34 = sc.integrate.quad(fec34, -sc.sqrt(R0**2-Y1**2), X1-r0 )

#     def fec25(x):
#         y = Y2
#         n1 = [[0.],[1.],[0.]]
        
#         s1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[1] 
            
#         ec = (s1 @ n1).T @ b2 / 2.0
        
#         return ec
#     ec25 = sc.integrate.quad(fec25, X2+r0, sc.sqrt(R0**2-Y2**2)  )

#     def fec45(x):
#         y = Y2
#         n1 = [[0.],[1.],[0.]]
        
#         s2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[1] 
            
#         ec = (s2 @ n1).T @ b2 / 2.0
        
#         return ec
#     ec45 = sc.integrate.quad(fec45, X2+r0, sc.sqrt(R0**2-Y2**2)  )



#     # Ec  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#     loc = locals()
#     Ec = np.zeros([4,5])
#     for i in np.arange(1,5,1):
#         for j in np.arange(1,4,1):
#             ec_2 = (loc['ec'+str(i)+str(j)])
#             Ec[i-1,j-1] = ec_2[0]
#     Ec[0,3] = ec14[0]
#     Ec[2,3] = ec34[0]
#     Ec[1,4] = ec25[0]
#     Ec[3,4] = ec45[0]

# #    return Ec#,u1,u2
    

    
#     def delta_disp(coor):
# #        u1 = stroh_Stroh_u0_s0(p, A, B, x, y, b1, X1, Y1, cut1)[0]
# #        u2 = stroh_Stroh_u0_s0(p, A, B, x, y, b2, X2, Y2, cut2)[0]
#         disptot = np.empty(shape=(coor.shape[0],3))
#         disp1 = np.empty(shape=(coor.shape[0],3))
#         disp2 = np.empty(shape=(coor.shape[0],3))
#         for i in np.arange(coor.shape[0]):
#             disp1[i] = stroh_Stroh_u0_s0(p, A, B,coor[i,0],coor[i,1], b1, X1, Y1, cut1)[0].T
#             disp2[i] = stroh_Stroh_u0_s0(p, A, B,coor[i,0],coor[i,1], b2, X2, Y2, cut2)[0].T
#         disptot = disp1 + disp2
#         return disptot        
#     disptot = delta_disp(coor)
#     return Ec, disptot




