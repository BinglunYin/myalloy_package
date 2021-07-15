
import numpy as np
from myalloy import calc_elastic_constant as cec 




def calc_yield_strength(self, param={}):
    if not hasattr(self, 'V0'):
        self.calc_V0_from_a()
    V0 = self.V0

    if not hasattr(self, 'delta'):
        self.calc_delta()
    delta = self.delta


    if 'model_type' in param: 
        model_type = param['model_type']
    else:                     
        model_type = 'aniso'

    if (model_type == 'aniso') and (self.brav_latt == 'fcc'):
        print('==>  applying fcc ANISOtropic model, sigmay [MPa]')
        self.calc_Cijavg_from_Cij()
        A = self.Cijavg['A']
        mu111 = self.Cijavg['mu_111']  
        muV = self.Cijavg['mu_V']
        nuV = self.Cijavg['nu_V']
    elif model_type == 'iso':
        print('==>  applying ISOtropic model, sigmay [MPa]')
        A = 1
        mu111 = self.poly['mu']
        muV   = self.poly['mu']
        nuV   = self.poly['nu']
    
    if 'A' in param: 
        A = param['A']
    

    if self.brav_latt == 'fcc':
        At = 0.04865* (1- (A-1)/40 )
        AE = 2.5785 * (1- (A-1)/80 )
        alpha = 0.125
        b = (V0*4)**(1/3) / np.sqrt(2)
    elif self.brav_latt == 'bcc':
        At = 0.040 * (16/3)**(2/3)
        AE = 2.00  * (16/3)**(1/3)
        alpha = 0.0833
        b = (V0*2)**(1/3) *np.sqrt(3)/2
  

    if 'alpha' in param: 
        alpha = param['alpha']
    
    if 'et0' in param: 
        et0 = param['et0']
    else:              
        et0 = 1e4
    
    if 'T' in param: 
        T = param['T']
    else:            
        T = 300
   
    if 'et' in param: 
        et = param['et']
    else:             
        et = 1e-3
        

    #-------------------
    Gamma = alpha * mu111 * b**2
    P = muV*(1+nuV)/(1-nuV)
    ty0 = At *(Gamma/b**2)**(-1/3)       *P**(4/3) *delta**(4/3) *1000  # [MPa]
    dEb = AE *(Gamma/b**2)**( 1/3) *b**3 *P**(2/3) *delta**(2/3) *1e-21 # [J]
  
    ty = calc_ty(et0, T, et, ty0, dEb)
    sigmay = 3.06 * ty
    print(sigmay)

    #-------------------
    wc = (np.pi/(2**(5/2) -1))**(1/3) *(dEb**2/(Gamma*b*ty0))**(1/3) *1e15  # [Ang]
    zetac = np.pi *dEb /( 2 *wc *b *ty0) *1e24   # [Ang]
    qe = 1.602176634e-19 
    sigma_dUsd = ( (2**(5/2)-1)**(-1) * 4 ) *dEb /qe   # [eV]
    
    
    #-------------------
    if 'filename' in param: 
    
        filen = 'sst_' + param['filename'] + '.txt'
        f = open(filen,"w+")
        
        f.write('# solute strengthening theory: \n' )
        f.write('%16s %16s %16s %16s \n' \
        %('alpha', 'et0 (/s)', 'T (K)', 'et (/s)' ) )
        f.write('%16.8f %16.1e %16.1f %16.1e \n\n' \
        %(alpha, et0, T, et) )
    
        f.write('%16s %16s %16s \n' \
        %('Zener A', 'At', 'AE') )
        f.write('%16.8f %16.8f %16.8f \n\n' \
        %(A, At, AE) )
        
        f.write('%16s %16s %16s \n' \
        %('b (Ang)', 'a_fcc (Ang)', 'a_bcc (Ang)') )
        f.write('%16.8f %16.8f %16.8f \n\n' \
        %(b, b*np.sqrt(2), b*2/np.sqrt(3)) )
        
        f.write('%16s %16s \n' \
        %('delta*100', 'delta*V0*3') )
        f.write('%16.8f %16.8f \n\n' \
        %(delta*100, delta*V0*3) )

        f.write('%16s %16s %16s \n' \
        %('mu111 (GPa)', 'muV (GPa)', 'nuV') )
        f.write('%16.1f %16.1f %16.4f \n\n' \
        %(mu111, muV, nuV) )
        
        f.write('%16s %16s \n' \
        %('ty0 (MPa)', 'dEb (eV)' ) )
        f.write('%16.4f %16.8f \n\n' \
        %(ty0, dEb/qe) )

        f.write('%16s %16s \n' \
        %('ty (MPa)', 'sigmay (MPa)' ) )
        f.write('%16.4f %16.4f \n\n' \
        %(ty, sigmay) )

        f.write('%16s %16s %16s %16s \n' \
        %('wc (Ang)', 'wc/b', 'zetac (Ang)', 'zetac/b' ) )
        f.write('%16.4f %16.4f %16.4f %16.4f \n\n' \
        %(wc, wc/b, zetac, zetac/b) )

        f.write('%16s \n' \
        %('sigma_dUsd (eV)' ) )
        f.write('%16.4f \n\n' \
        %(sigma_dUsd) )



        if hasattr(self, 'EPI'):

            from myalloy import solute_strengthening_theory_EPI as sstEPI 

            sigma_dUss = sstEPI.calc_sigma_dUss(self, b, wc, zetac, t='fcc_partial')


            sigma_dUtot = np.sqrt( sigma_dUsd**2 + sigma_dUss**2 )
            sigma_ratio = sigma_dUtot / sigma_dUsd


            sigma_ratio = param['sigma_ratio']
            f.write('\n# Add solute-solute interaction: \n' )

            f.write('%16s %16s \n' \
            %('sigma_dUss', 'sigma_dUtot') )
            f.write('%16.4f %16.4f \n\n' \
            %(sigma_dUss, sigma_dUtot) )

            f.write('%16s %16s %16s \n' \
            %('sigma_ratio', 'ratio**(4/3)', 'ratio**(2/3)') )
            f.write('%16.4f %16.4f %16.4f \n\n' \
            %(sigma_ratio, sigma_ratio**(4/3), sigma_ratio**(2/3)) )

            ty0_tot = ty0*sigma_ratio**(4/3)
            dEb_tot = dEb*sigma_ratio**(2/3)
            sigmay_tot = 3.06*calc_ty(et0, T, et, ty0_tot, dEb_tot)

            f.write('%16s %16s %16s \n' \
            %('ty0_tot (MPa)', 'dEb_tot (eV)', 'sigmay_tot (MPa)' ) )
            f.write('%16.4f %16.4f %16.4f \n\n' \
            %(ty0_tot, dEb_tot/qe, sigmay_tot) )






        f.write('\n\n# alloy properties: \n')

        if hasattr(self, 'V0'):
            f.write('\n%16s \n' \
            %('V0_alloy (Ang^3)') )
            f.write('%16.8f \n\n' \
            %(self.V0) )
        
        if hasattr(self, 'dV'):
            f.write('%16s %16s %16s \n' \
            %('cn', 'dV (Ang^3)', 'Velem (Ang^3)') )
            for i in np.arange(self.nelem):
                f.write('%16.8f %16.8f %16.8f \n' \
                %(self.cn[i], self.dV[i], self.dV[i]+self.V0) )
             
            f.write(' \n')
            f.write('%16s \n' \
            %('cn @ dV') )
            f.write('%16.8f \n\n' \
            %(self.cn @ self.dV) )
        
        if hasattr(self, 'Cij'):
            f.write('%16s \n' \
            %('Cij_alloy (GPa)') )
            f.write('%16.1f %16.1f %16.1f \n\n' \
            %(self.Cij[0], self.Cij[1], self.Cij[2]) )

        if hasattr(self, 'Cijavg'):
            f.write(' Cij average: B, mu, nu, E \n')
            B  = self.Cijavg['B_V']
            mu = self.Cijavg['mu_V']
            nu = self.Cijavg['nu_V']
            f.write('%10s %16.1f %16.1f %16.4f %16.1f \n' \
            %('Voigt:', B, mu, nu, cec.calc_E_from_mu_nu(mu, nu)  ))

            B  = self.Cijavg['B_R']
            mu = self.Cijavg['mu_R']
            nu = self.Cijavg['nu_R']
            f.write('%10s %16.1f %16.1f %16.4f %16.1f \n' \
            %('Reuss:', B, mu, nu, cec.calc_E_from_mu_nu(mu, nu)  ))

            B  = self.Cijavg['B_H']
            mu = self.Cijavg['mu_H']
            nu = self.Cijavg['nu_H']
            f.write('%10s %16.1f %16.1f %16.4f %16.1f \n\n' \
            %('Hill:', B, mu, nu, cec.calc_E_from_mu_nu(mu, nu)  ))

        if hasattr(self, 'polyelem'):
            f.write(' elemental poly: \n')
            f.write('%16s %16s \n' \
            %('mu (GPa)', 'nu') )
            for i in np.arange(self.nelem):
                f.write('%16.1f %16.4f \n' \
                %(self.polyelem[i,0], self.polyelem[i,1]) )
            f.write(' \n')

        if hasattr(self, 'Cijelem'):
            f.write(' elemental Cij: \n')
            for i in np.arange(self.nelem):
                f.write('%16.1f %16.1f %16.1f \n' \
                %(self.Cijelem[i,0], self.Cijelem[i,1], self.Cijelem[i,2]) )
            f.write(' \n')
            
        f.close() 





#==============================


def calc_ty(et0, T, et, ty0, dEb):
    k = 1.380649e-23
    ty = ty0 *(1 - (k*T/dEb *np.log(et0/et))**(2/3) )  # [MPa]
    return ty












#==============================

def fcc_Vegard_strength( ROMtype, cn, param = {}):
    from myalloy import main 
    from myalloy import solute_strengthening_theory_database as sstb 

    if ROMtype == 'polyelem':
        data1 = sstb.fcc_elem_poly()
    elif ROMtype == 'Cijelem':
        data1 = sstb.fcc_elem_Cij()

    n0 = data1.shape[0] - cn.shape[0]
    cn = np.append(cn, np.zeros(n0) )

    alloy1 = main.alloy_class('fcc_Vegard_strength', cn)
    alloy1.Velem = data1[:,0]
    alloy1.calc_from_Velem()

    if ROMtype == 'polyelem':
        alloy1.polyelem = data1[:, 1:3]
        alloy1.calc_from_polyelem()
        param.update({'model_type': 'iso'})

    elif ROMtype == 'Cijelem':
        alloy1.Cijelem = data1[:, 1:4]
        alloy1.calc_from_Cijelem()
        param.update({'model_type': 'aniso'})

    alloy1.calc_yield_strength(param)





