
# By B. Yin
# 2021.07.20

# This function creates initial dissociated dislocation 
# The SF plane is in the x1-x3 plane 
# cut1 = cut2 = 0 
# Energy integral is not possible due to the singularity




import numpy as np
import sys 


def calc_stroh_2(self, slip_system='basal_a_edge', param={}):
    
    qe=1.6021766208e-19   #  :??

    #--------------------------------------------
    # pre    
    #--------------------------------------------

    if not hasattr(self, 'c'):
        self.c = self.a

    from myalloy import stroh_dislocations_slip_system as ss
    mm, theta, b1, b2 = \
        ss.slip_system(self, slip_system=slip_system, param=param)


    # rotate x1-x2 to make theta=0

    temp = mm.copy() 
    temp[0,:] =  mm[0,:] * np.cos(theta) + mm[1,:] * np.sin(theta) 
    temp[1,:] = -mm[0,:] * np.sin(theta) + mm[1,:] * np.cos(theta) 
    mm = temp.copy()


    # express b1, b2 in the new mm 

    R = np.array([ 
        [ np.cos(theta), np.sin(theta),  0],
        [-np.sin(theta), np.cos(theta),  0],
        [             0,             0,  1],
    ])
    print(R)

    b1 = R @ b1
    b2 = R @ b2

    theta = 0 





    #==============================

    a   = self.a
    c   = self.c
    Cij = self.Cij 
    brav_latt = self.brav_latt
    gamma = self.gamma *1.0e-3 *(1/qe/1e20)  # to [eV/Ang^2]
    
    r0 = 0.5*np.linalg.norm(b1+b2)
    R0 = 1e7*a
       

    from myalloy import calc_elastic_constant as cec 
    CIJ, CIJKL, CIJKL2, CIJ2 = cec.rotate_Cij(brav_latt, Cij, mm)

    CIJKL2 = CIJKL2 *1.0e9 *(1.0/qe/1.0e30)  # to [eV/Ang^3]


    #--------------------------------------------
    # p, A, B; K, E_r, E_theta; u, s
    #--------------------------------------------

    from myalloy import stroh_dislocations_formalism as fm 
    N, p, A, B = fm.calc_N_p_A_B(CIJKL2)
    K1, K2, K12 = fm.calc_K(b1, b2, B) 
    
    if 'r12' in param: 
        r12 = param['r12']
    else:
        r12 = K12/gamma

    if r12 < 2.0*r0 :
        print(r12, 2.0*r0)
        sys.exit('ABORT: r12<2*r0.')



    Er, Et = fm.calc_Er_Et(theta, b1, b2, r0,   p, B, K12, r12)



    X1 = -r12/2
    Y1 = 0
    X2 =  r12/2
    Y2 = 0
    cut1 = 0
    cut2 = 0 
    
    from functools import partial
    stroh_u1s1 = partial(fm.stroh_u0_s0, p=p, A=A, B=B, \
        b=b1, X=X1, Y=Y1, cut=cut1)
    
    stroh_u2s2 = partial(fm.stroh_u0_s0, p=p, A=A, B=B, \
        b=b2, X=X2, Y=Y2, cut=cut2)




    if 'pos_in' in param: 
        fm.calc_pos_out(stroh_u1s1, stroh_u2s2, param['pos_in'])

    
    


    if 'output_name' in param: 
        from myalloy import stroh_dislocations as sd 

        Ec = np.array([
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0], 
        ])

        sd.write_output(param['output_name'], qe, slip_system, \
            a, c, r0, R0, theta, b1, b2, gamma, \
            K1, K2, K12, r12, X1, Y1, X2, Y2, Er, Et, Ec, \
            mm, CIJ, CIJ2, N, p, A, B )





