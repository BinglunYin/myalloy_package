
import numpy as np
from myalloy import calc_elastic_constant as cec 




def calc_yield_strength(self, param={}):
  
    self.calc_b_from_V0()
    b = self.b 
    V0 = self.V0


    
    if not hasattr(self, 'delta'):
        self.calc_delta_from_dV()
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

    elif self.brav_latt == 'bcc':
        At = 0.040 * (16/3)**(2/3)
        AE = 2.00  * (16/3)**(1/3)
        alpha = 0.0833
  

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
    
    ty0, dEb, wc, zetac, sigma_dUsd, dEpsd = calc_model(alpha, b, mu111, muV, nuV, delta, At, AE)

    ty0   = ty0   *1e-6
    wc    = wc    *1e10
    zetac = zetac *1e10
    sigma_dUsd = sigma_dUsd / self.qe
    dEpsd      = dEpsd / self.qe


    sigmay = calc_sigmay(et0, T, et, ty0, dEb)
    print(sigmay)


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
        %(ty0, dEb/self.qe) )

        f.write('%16s \n' \
        %('sigmay (MPa)' ) )
        f.write('%16.4f \n\n' \
        %(sigmay) )

        f.write('%16s %16s %16s %16s \n' \
        %('wc (Ang)', 'wc/b', 'zetac (Ang)', 'zetac/b' ) )
        f.write('%16.4f %16.4f %16.4f %16.4f \n\n' \
        %(wc, wc/b, zetac, zetac/b) )

        f.write('%16s %16s \n' \
        %('sigma_dUsd (eV)', 'dEpsd (eV)' ) )
        f.write('%16.4f %16.4f \n\n' \
        %(sigma_dUsd, dEpsd) )



        if hasattr(self, 'EPI'):

            from myalloy import solute_strengthening_theory_EPI as sstEPI 

            sigma_dUss_tilde_f = sstEPI.calc_sigma_dUss_tilde(self,  t='fcc_full')    
            sigma_dUss_tilde_p = sstEPI.calc_sigma_dUss_tilde(self,  t='fcc_partial')   


            dEpss = sstEPI.calc_dEpss(self, wc, t='fcc_partial')    # [eV]

            dEp_tot = np.sqrt( dEpsd**2 + dEpss**2 )
            dEp_ratio = dEp_tot / dEpsd 

            ty0_tot = ty0 * dEp_ratio**(4/3)
            dEb_tot = dEb * dEp_ratio**(2/3)
            sigmay_tot = calc_sigmay(et0, T, et, ty0_tot, dEb_tot)


            f.write('\n# With solute-solute interaction strengthening in ideal random alloy: \n' )
            f.write('# - slip: \n' )

            f.write('%30s %30s \n' \
            %('sigma_dUss_tilde_f (eV)', 'sigma_dUss_tilde_p (eV)' ) )
            f.write('%30.4f %30.4f \n\n' \
            %(sigma_dUss_tilde_f, sigma_dUss_tilde_p) )         


            f.write('# - strengthening: \n' )

            f.write('%16s %16s %16s \n' \
            %('dEpss (eV)', 'dEp_tot (eV)', 'dEp_ratio' ) )
            f.write('%16.4f %16.4f %16.4f \n\n' \
            %(dEpss, dEp_tot, dEp_ratio) )         

            f.write('%16s %16s %16s \n' \
            %('ty0_tot (MPa)', 'dEb_tot (eV)', 'sigmay_tot (MPa)' ) )
            f.write('%16.4f %16.4f %16.4f \n\n' \
            %(ty0_tot, dEb_tot/self.qe, sigmay_tot) )






        if hasattr(self, 'EPI') and hasattr(self, 'SRO'):

            from myalloy import solute_strengthening_theory_EPI as sstEPI 

            tauA = sstEPI.calc_tau_A(self, b, t='fcc_full')
            
            gamma_APB = tauA *1e6 *b *1e-10 *1e3


            f.write('\n# SRO average strengthening: \n' )

            f.write('%16s %33s \n' \
            %('tauA (MPa)', 'gamma_APB (mJ/m^2)') )
            f.write('%16.4f %33.4f \n\n' \
            %(tauA, gamma_APB) )







        f.write('\n# ====================\n# alloy properties: \n')
        
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



        with np.printoptions(linewidth=200, \
            precision=8, suppress=True):

            if hasattr(self, 'V0'):
                f.write('V0 \n')
                f.write(str(self.V0)+'\n\n')

            if hasattr(self, 'Cij'):
                f.write('Cij (GPa) \n')
                f.write(str(self.Cij)+'\n\n')

            if hasattr(self, 'polyelem'):
                f.write('polyelem: mu (GPa), nu \n')
                f.write(str(self.polyelem)+'\n\n')

            if hasattr(self, 'Cijelem'):
                f.write('Cijelem: elemental Cij: \n')
                f.write(str(self.Cijelem)+'\n\n')

            if hasattr(self, 'EPI'):
                f.write('EPI (eV): \n')
                f.write(str(self.EPI)+'\n\n')

            if hasattr(self, 'SRO'):
                f.write('SRO: \n')
                f.write(str(self.SRO)+'\n\n')

        f.close() 




    return sigmay








#==============================




def calc_model(alpha, b, mu111, muV, nuV, delta, At, AE):
    from myvasp import vasp_func as vf 
    
    b     = b*1e-10
    mu111 = mu111*1e9
    muV   = muV*1e9

    Gamma = alpha * mu111 * b**2    
    P = muV*(1+nuV)/(1-nuV)  

    ty0 = At *(Gamma/b**2)**(-1/3)       *P**(4/3) *delta**(4/3) 
    dEb = AE *(Gamma/b**2)**( 1/3) *b**3 *P**(2/3) *delta**(2/3) 

    #-------------------
    wc = (np.pi/(2**(5/2) -1))**(1/3) * (dEb**2/(Gamma*b*ty0))**(1/3) 
    zetac = np.pi *dEb /(2 *wc *b *ty0) 
    sigma_dUsd = ( 4 / (2**(5/2)-1) ) *dEb      
    dEpsd = ( zetac / (np.sqrt(3)*b) )**(-1/2) * sigma_dUsd 
    
    #-------------------
    kzeta = (4*np.sqrt(3))**(1/3)
    k101  = np.pi * 2**(-10/3) * 3**(-1/3) * (2**(5/2) -1)
    k122  =         2**(-5/3)  * 3**(-1/6) * (2**(5/2) -1)
    
    vf.confirm_0( kzeta * ( Gamma**( 2) * b       * wc**( 4) * dEpsd**(-2) )**(1/3) / zetac -1, str1='zetac' ) 
    vf.confirm_0( k101  * ( Gamma**(-1) * b**(-5) * wc**(-5) * dEpsd**( 4) )**(1/3) / ty0   -1, str1='ty0'   ) 
    vf.confirm_0( k122  * ( Gamma       * b**(-1) * wc**( 2) * dEpsd**( 2) )**(1/3) / dEb   -1, str1='dEb'   ) 

    return ty0, dEb, wc, zetac, sigma_dUsd, dEpsd 





def calc_sigmay(et0, T, et, ty0, dEb):
    k = 1.380649e-23
    sigmay = 3.06* ty0 *(1 - (k*T/dEb *np.log(et0/et))**(2/3) )  # [MPa]
    return sigmay  



















#==============================



def calc_yield_strength_et_T(self):

    T_list  = np.array([77, 300, 500])
    et_list = np.arange(-10, 3, 1)

    sigmay_all = np.zeros( [len(T_list), len(et_list)] )

    for i in np.arange(len(T_list)):
        for j in np.arange(len(et_list)):

            sigmay_all[i, j] = self.calc_yield_strength(
                param = { 'T': T_list[i], 'et': 10.0**(et_list[j]) } )
        
    print('==> sigmay:', sigmay_all)



    from myvasp import vasp_func as vf 
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig_wh = [3.15, 3]
    fig_subp = [1, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)

    fig_pos  = np.array([0.20, 0.16, 0.75, 0.8])
    ax1.set_position(fig_pos)
   
    xi = 10.0**et_list

    ax1.set_xscale('log')

    for i in np.arange( len(T_list) ):
        str1 = '%d K' %(T_list[i])
        ax1.plot(xi, sigmay_all[i,:], '-o', label = str1)
        
    ax1.legend(loc='best')  


    [ymin, ymax] = ax1.get_ylim()
    ax1.set_ylim([0, ymax])        


    ax1.set_xlabel('Loading strain rate $\\dot{\\epsilon}$ (s$^{-1}$)')        
    ax1.set_ylabel('Yield strength $\\sigma_y$ (MPa)')        

        
    
    plt.savefig('fig_sigmay_et_T.pdf')
    plt.close('all')


















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





