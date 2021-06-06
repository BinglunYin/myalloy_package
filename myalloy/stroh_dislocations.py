
# By T. Liu and B. Yin
# 2021.06.04


import numpy as np
import sys 


def calc_stroh(self, slip_system='basal_a_edge', param={}):
    
    qe=1.6021766208e-19   #  :??

    #--------------------------------------------
    # pre    
    #--------------------------------------------

    if not hasattr(self, 'c'):
        self.c = self.a

    from myalloy import stroh_dislocations_slip_system as ss
    mm, theta, b1, b2 = \
        ss.slip_system(self, slip_system=slip_system, param=param)
    

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

    X1 = 0.
    Y1 = 0.
    X2 = X1+r12*np.cos(theta)
    Y2 = Y1+r12*np.sin(theta)
    cut1 = -np.pi
    cut2 = 0 
    
    from functools import partial
    stroh_u1s1 = partial(fm.stroh_u0_s0, p=p, A=A, B=B, \
        b=b1, X=X1, Y=Y1, cut=cut1)
    
    stroh_u2s2 = partial(fm.stroh_u0_s0, p=p, A=A, B=B, \
        b=b2, X=X2, Y=Y2, cut=cut2)

    
    if 'pos_in' in param: 
        fm.calc_pos_out(stroh_u1s1, stroh_u2s2, param['pos_in'])


    #--------------------------------------------
    # energy Ec 
    #--------------------------------------------

    from myalloy import stroh_dislocations_energy as en
    Ec = en.calc_Ec(stroh_u1s1, stroh_u2s2, r0, R0, \
        b1, b2, X1, Y1, X2, Y2, cut1, cut2)


    if 'output_name' in param: 
        write_output(param['output_name'], qe, slip_system, \
            a, c, r0, R0, theta, b1, b2, gamma, \
            K1, K2, K12, r12, X1, Y1, X2, Y2, Er, Et, Ec, \
            mm, CIJ, CIJ2, N, p, A, B )









def write_output(output_name, qe, slip_system, \
    a, c, r0, R0, theta, b1, b2, gamma, \
    K1, K2, K12, r12, X1, Y1, X2, Y2, Er, Et, Ec, \
    mm, CIJ, CIJ2, N, p, A, B ):


    filen = 'stroh_' + output_name + '.txt'
    f = open(filen,"w+")


    f.write('# stroh formalism for two dislocations: \n' )
    f.write('# slip system: %s \n\n' \
        %(slip_system) )


    f.write('%16s %16s %16s %16s \n' \
        %('a (Ang)', 'c/a', 'r0/a', 'R0/a' ) )
    f.write('%16.8f %16.8f %16.8f %16.4e \n\n' \
        %(a, c/a, r0/a, R0/a) )


    f.write('%16s %16s %16s %16s \n' \
        %('theta(rad)', 'theta(degree)', 'gamma(eV/Ang^2)', 'gamma(mJ/m^2)' ) )
    f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
        %(theta, theta/np.pi*180, gamma, gamma*(qe*1e3)/(1e-20)) )
  
            
    f.write('%16s %16s %16s \n' \
        %('b1 (Ang)', 'b2', 'b1+b2' ) )
    for i in np.arange(3):
        f.write('%16.8f %16.8f %16.8f \n' \
        %(b1[i], b2[i], b1[i]+b2[i]) )            
    f.write('\n')
    

    f.write('%16s %16s %16s \n' \
        %('K1 (eV/Ang)', 'K2', 'K12' ) )
    f.write('%16.8f %16.8f %16.8f \n\n' \
        %(K1, K2, K12) )


    f.write('%16s %16s %16s %16s \n' \
        %('r12 (Ang)', 'r12/a', 'r12/b', 'r12/r0' ) )
    f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
        %(r12, r12/a, r12/np.linalg.norm(b1+b2), r12/r0) )


    f.write('%16s %16s %16s %16s \n' \
        %('X1 (Ang)', 'Y1', 'X2', 'Y2' ) )
    f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
        %(X1, Y1, X2, Y2) )


    Esf = gamma*r12
    f.write('%16s %16s %16s \n' \
        %('E_r (eV/Ang)', 'E_theta', 'E_sf' ) )
    f.write('%16.8f %16.8f %16.8f \n\n' \
        %(Er, Et, Esf) )


    f.write('%16s %16s \n' \
        %('Ec.sum()', 'E_tot' ) )
    f.write('%16.8f %16.8f \n\n' \
        %(Ec.sum(), Ec.sum()+Esf ) )




    with np.printoptions(linewidth=200, \
        precision=8, suppress=True):

        f.write('Energy contribution, Ec (eV/Ang) \n')
        f.write(str(Ec)+'\n\n')

        f.write('np.sum(Ec, axis=0) \n')
        f.write(str(np.sum(Ec, axis=0))+'\n\n')
        
        f.write('np.sum(Ec, axis=1) \n')
        f.write(str(np.sum(Ec, axis=1))+'\n\n')
                
                
    
        f.write('\n\nmm\n')
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





