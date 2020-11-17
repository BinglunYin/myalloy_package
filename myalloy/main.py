
import numpy as np
import sys 


class alloy_class:
    def __init__(self, name, cn, brav_latt = 'fcc'):
        self.name = name
        self.cn = cn / cn.sum()
        self.brav_latt = brav_latt

        self.nelem = self.cn.shape[0]



    # print
    def print_alloy(self):
        print(self.name, self.cn)



    def dump(self):
        for attr in dir(self):
            if hasattr(self, attr):
                print('self.%s = %s' % (attr, getattr(self, attr)))



    # volume
    def calc_V0_from_a(self):
        if hasattr(self, 'V0'):
            sys.exit('ABORT: V0 exists. Double check!')
        else:
            if self.brav_latt is 'fcc':
                self.V0 = self.a **3/4 
            elif self.brav_latt is 'bcc':
                self.V0 = self.a **3/2 



    def calc_from_Velem(self):
        self.V0 = self.cn @ self.Velem
        self.dV = self.Velem - self.V0



    def calc_delta(self):
        if not hasattr(self, 'V0'):
            self.calc_V0_from_a()
        self.delta = np.sqrt( self.cn @ np.square(self.dV) ) /self.V0/3



    # elasticity
    def calc_Cijavg_from_Cij(self):      
        self.Cijavg = calc_Cijavg_from_Cij(self.Cij)



    def calc_from_polyelem(self):
        
        Belem = np.array([]) 
        Eelem = np.array([])   

        for i in np.arange(self.nelem):
            Belem = np.append(Belem, \
            calc_B_from_mu_nu(self.polyelem[i,0], self.polyelem[i,1]) )

            Eelem = np.append(Eelem, \
            calc_E_from_mu_nu(self.polyelem[i,0], self.polyelem[i,1]) )

        mu = self.cn @ self.polyelem[:,0]

        # B  = self.cn @ Belem
        # nu = self.calc_nu_from_B_mu(B, mu)

        E  = self.cn @ Eelem
        nu = calc_nu_from_E_mu(E, mu)

        self.poly = {'mu':mu, 'nu':nu}



    def calc_from_Cijelem(self):
        self.Cij = self.cn @ self.Cijelem
        


    # solute strengthening theory
    def calc_yield_strength(self, param={}):
        from myalloy import solute_strengthening_theory as sst 
        sst.calc_yield_strength(self, param=param)




















#==============================
# functions
#==============================

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




def calc_Cijavg_from_Cij(Cij, brav_latt='fcc'):
    Cijavg={}
    if (brav_latt is 'fcc') or (brav_latt is 'bcc'):
        [C11, C12, C44] = Cij
        # fcc slip
        mu_111 = C44 - ( 2*C44 +C12 -C11 )/3
        Cijavg['mu_111'] = mu_111
        CIJ=np.array([
            [C11, C12, C12,   0,   0,   0],
            [  0, C11, C12,   0,   0,   0],
            [  0,   0, C11,   0,   0,   0],
            [  0,   0,   0, C44,   0,   0],
            [  0,   0,   0,   0, C44,   0],
            [  0,   0,   0,   0,   0, C44],
        ])
        CIJ = symmetrize_matrix(CIJ)
    
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



