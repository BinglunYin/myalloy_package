
import numpy as np
import sys 
from myalloy import calc_elastic_constant as cec




class alloy_class:
    
    qe = 1.602176634e-19


    def __init__(self, name, cn, brav_latt = 'fcc'):
        self.name = name
        self.cn = cn / cn.sum()
        self.brav_latt = brav_latt

        self.nelem = self.cn.shape[0]



    def print_attributes(self):
        for attr in dir(self):
            print('%s:\n%s \n' %(attr, getattr(self, attr)) )






    # volume
    def calc_V0_from_a(self):
        if hasattr(self, 'V0'):
            sys.exit('ABORT: V0 exists. Double check!')
        else:
            if self.brav_latt == 'fcc':
                self.V0 = self.a **3/4 
            elif self.brav_latt == 'bcc':
                self.V0 = self.a **3/2 



    def calc_b_from_V0(self):
        if not hasattr(self, 'V0'):
            self.calc_V0_from_a()

        if self.brav_latt == 'fcc':
            self.b = (self.V0*4)**(1/3) / np.sqrt(2)
        elif self.brav_latt == 'bcc':
            self.b = (self.V0*2)**(1/3) *np.sqrt(3)/2



    def calc_from_Velem(self):
        self.V0 = self.cn @ self.Velem
        self.dV = self.Velem - self.V0



    def calc_delta_from_dV(self):
        if not hasattr(self, 'V0'):
            self.calc_V0_from_a()
        self.delta = np.sqrt( self.cn @ np.square(self.dV) ) /self.V0/3






    # elasticity
    def calc_Cijavg_from_Cij(self):      
        self.Cijavg = cec.calc_Cijavg_from_Cij(self.brav_latt, self.Cij)



    def calc_from_polyelem(self):
        
        Belem = np.array([]) 
        Eelem = np.array([])   

        for i in np.arange(self.nelem):
            Belem = np.append(Belem, \
                cec.calc_B_from_mu_nu(self.polyelem[i,0], self.polyelem[i,1]) )

            Eelem = np.append(Eelem, \
                cec.calc_E_from_mu_nu(self.polyelem[i,0], self.polyelem[i,1]) )

        mu = self.cn @ self.polyelem[:,0]

        # B  = self.cn @ Belem
        # nu = self.calc_nu_from_B_mu(B, mu)

        E  = self.cn @ Eelem
        nu = cec.calc_nu_from_E_mu(E, mu)

        self.poly = {'mu':mu, 'nu':nu}



    def calc_from_Cijelem(self):
        self.Cij = self.cn @ self.Cijelem






    # get EPI
    def get_EPI_from_file(self, filename='y_post_EPI.beta2_4.txt'):
        beta2 = np.loadtxt(filename)
        self.set_EPI(beta2) 



    def set_EPI(self, beta2):
        EPI, shellmax = EPI_reshape(self.nelem, beta2)
        self.EPI = EPI 
        self.shellmax = shellmax 



    # get SRO
    def get_SRO_from_file(self, filename='y_post_WC_SRO_shell_avg.txt'):
        data = np.loadtxt(filename)
        SRO, shellmax = EPI_reshape(self.nelem, data)
        self.SRO = SRO






    # solute strengthening theory
    def calc_yield_strength(self, param={}):
        from myalloy import solute_strengthening_theory as sst 
        sigmay = sst.calc_yield_strength(self, param=param)
        return sigmay 
        


    def calc_yield_strength_et_T(self):
        from myalloy import solute_strengthening_theory as sst 
        sst.calc_yield_strength_et_T(self)










   # slip 
    def calc_std_gamma_APB(self, l1, l2, param={}):
        from myalloy import solute_strengthening_theory_EPI as sstEPI 
        sigma_dUss_f = sstEPI.calc_std_gamma_APB(self, l1, l2, param=param)
        return sigma_dUss_f 





    # Stroh's formalism
    def calc_stroh(self, slip_system='basal_a_edge', param={}):
        from myalloy import stroh_dislocations as stroh 
        stroh.calc_stroh(self, \
            slip_system=slip_system, param=param)


    def calc_stroh_2(self, slip_system='basal_a_edge', param={}):
        from myalloy import stroh_dislocations_2 as stroh_2 
        stroh_2.calc_stroh_2(self, \
            slip_system=slip_system, param=param)










def EPI_reshape(nelem, A):
    shellmax = len(A)/ (nelem*(nelem-1)/2) 

    if np.abs(shellmax - int(shellmax)) > 1e-10:
        sys.exit('ABORT: wrong shellmax.')
    else:
        shellmax = int(shellmax)

    temp = int(len(A)/shellmax)        
    A2 = A.reshape(shellmax, temp)

    A3 = np.zeros((shellmax, nelem, nelem))
    for i in np.arange(shellmax):
        m = -1
        for j in np.arange(nelem-1):
            for k in np.arange(j+1, nelem):
                m = m+1  
                A3[i, j, k] = A2[i, m]
                A3[i, k, j] = A2[i, m]
    
    return A3, shellmax  





