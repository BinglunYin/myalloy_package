
import numpy as np
import sys 


class alloy_class:
    def __init__(self, name, cn, brav_latt = 'fcc'):
        self.name = name
        self.cn = cn / cn.sum()
        self.brav_latt = brav_latt


    # print
    def print_alloy(self):
        print(self.name, self.cn)


    def dump(self):
        for attr in dir(self):
            if hasattr(self, attr):
                print('self.%s = %s' % (attr, getattr(self, attr)))


    # volume
    def calc_V0_from_a(self):
        if self.brav_latt is 'fcc':
            self.V0 = self.a **3/4 
        elif self.brav_latt is 'bcc':
            self.V0 = self.a **3/2 


    def calc_from_Vapp(self):
        self.V0 = self.cn @ self.Vapp
        self.dV = self.Vapp - self.V0


    def calc_delta(self):
        if not hasattr(self, 'V0'):
            self.calc_V0_from_a()
        self.delta = np.sqrt( self.cn @ np.square(self.dV) ) /self.V0/3


    # elasticity
    def calc_nu_from_B_mu(self, B, mu):
        nu = (3*B -2*mu)/(3*B +mu)/2
        return nu


    def calc_modulus(self):
        self.modulus={}

        if (self.brav_latt is 'fcc') or (self.brav_latt is 'bcc'):
            [C11, C12, C44] = self.Cij

            # fcc slip
            mu_111 = C44 - ( 2*C44 +C12 -C11 )/3
            self.modulus['mu_111'] = mu_111

            CIJ=np.array([
                [C11, C12, C12, 0, 0, 0],
                [C12, C11, C12, 0, 0, 0],
                [C12, C12, C11, 0, 0, 0],
                [0, 0, 0, C44, 0, 0],
                [0, 0, 0, 0, C44, 0],
                [0, 0, 0, 0, 0, C44],
            ])

        # Zener anisotropy
        self.modulus['A'] = 2*CIJ[3,3]/(CIJ[0,0]-CIJ[0,1])

        # Voigt       
        c1 = CIJ[0,0] +CIJ[1,1] +CIJ[2,2]
        c2 = CIJ[0,1] +CIJ[0,2] +CIJ[1,2]
        c3 = CIJ[3,3] +CIJ[4,4] +CIJ[5,5]
        
        B_V  = (c1 +2*c2)/9
        mu_V = (c1 -c2 +3*c3)/15
        nu_V = self.calc_nu_from_B_mu(B_V, mu_V)
        
        # Reuss
        SIJ = np.linalg.inv(CIJ)

        s1 = SIJ[0,0] +SIJ[1,1] +SIJ[2,2]
        s2 = SIJ[0,1] +SIJ[0,2] +SIJ[1,2]
        s3 = SIJ[3,3] +SIJ[4,4] +SIJ[5,5]

        B_R  = 1/(s1 +2*s2)
        mu_R = 15/(4*s1 -4*s2 +3*s3)
        nu_R = self.calc_nu_from_B_mu(B_R, mu_R)
        
        # Hill
        B_H  = (B_V + B_R)/2
        mu_H = (mu_V + mu_R)/2
        nu_H = self.calc_nu_from_B_mu(B_H, mu_H)
        
        self.modulus.update({ \
            'B_V': B_V, 'mu_V': mu_V, 'nu_V': nu_V, \
            'B_R': B_R, 'mu_R': mu_R, 'nu_R': nu_R, \
            'B_H': B_H, 'mu_H': mu_H, 'nu_H': nu_H, \
        })
 


    # solute strengthening theory
    def calc_yield_strength(self, param={}):
        if hasattr(self, 'a'):
            self.calc_V0_from_a()

        if not hasattr(self, 'delta'):
            self.calc_delta()
        delta = self.delta


        if 'model_type' in param: model_type = param['model_type']
        else:                     model_type = 'aniso'

        if (model_type is 'aniso') and (self.brav_latt is 'fcc'):
            print('==>  applying fcc ANISOtropic model, sigmay [MPa]')
            self.calc_modulus()

            A = self.modulus['A']
            mu111 = self.modulus['mu_111']  
            muV = self.modulus['mu_V']
            nuV = self.modulus['nu_V']
        
        elif model_type is 'iso':
            print('==>  applying ISOtropic model, sigmay [MPa]')

            A = 1
            mu111 = self.poly['mu']
            muV   = self.poly['mu']
            nuV   = self.poly['nu']


        if self.brav_latt is 'fcc':
            At = 0.04865* (1- (A-1)/40 )
            AE = 2.5785 * (1- (A-1)/80 )
            alpha = 0.125
            b = (self.V0*4)**(1/3) / np.sqrt(2)

        elif self.brav_latt is 'bcc':
            At = 0.040 * (16/3)**(2/3)
            AE = 2.00  * (16/3)**(1/3)
            alpha = 0.0833
            b = (self.V0*2)**(1/3) *np.sqrt(3)/2


        if 'et0' in param: et0 = param['et0']
        else:              et0 = 1e4
        
        if 'et' in param: et = param['et']
        else:             et = 1e-3
        
        if 'T' in param: T = param['T']
        else:            T = 300

        #-------------------
        Gamma = alpha * mu111 * b**2
        P = muV*(1+nuV)/(1-nuV)

        ty0 = At *(Gamma/b**2)**(-1/3)       *P**(4/3) *delta**(4/3) *1000  # [MPa]
        dEb = AE *(Gamma/b**2)**( 1/3) *b**3 *P**(2/3) *delta**(2/3) *1e-21 # [J]

        k = 1.380649e-23
        ty = ty0 *(1 - (k*T/dEb *np.log(et0/et))**(2/3) )  # [MPa]
        sigmay = 3.06 * ty
        print(sigmay)

        #-------------------
        if 'filename' in param: 
        
            filen = 'sst_' + param['filename'] + '.txt'
            f = open(filen,"w+")

            f.write('# solute strengthening theory: \n' )

            f.write('%16s %16s %16s %16s \n' \
            %('alpha', 'et0 (/s)', 'T (K)', 'et (/s)' ) )

            f.write('%16.8f %16.1E %16.1f %16.1E \n\n' \
            %(alpha, et0, T, et) )

        
            f.write('%16s %16s %16s \n' \
            %('A', 'At', 'AE') )

            f.write('%16.8f %16.8f %16.8f \n\n' \
            %(A, At, AE) )


            f.write('%16s %16s %16s %16s \n' \
            %('b (Ang)', 'a_fcc (Ang)', 'a_bcc (Ang)', 'delta*100') )

            f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
            %(b, b*np.sqrt(2), b*2/np.sqrt(3), delta*100) )

  
            f.write('%16s %16s %16s \n' \
            %('mu111 (GPa)', 'muV (GPa)', 'nuV') )

            f.write('%16.1f %16.1f %16.4f \n\n' \
            %(mu111, muV, nuV) )


            f.write('%16s %16s \n' \
            %('ty0 (MPa)', 'dEb (eV)' ) )

            qe = 1.602176634e-19 
            f.write('%16.4f %16.8f \n\n' \
            %(ty0, dEb/qe) )


            f.write('%16s %16s \n' \
            %('ty (MPa)', 'sigmay (MPa)' ) )

            f.write('%16.4f %16.4f \n\n' \
            %(ty, sigmay) )


            f.close() 




