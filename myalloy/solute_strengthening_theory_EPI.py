
import numpy as np
from myalloy import calc_elastic_constant as cec 


# solute-solute interaction in random alloys
# https://doi.org/10.1016/j.actamat.2020.08.011


def load_Theta_fcc_partial():
    Theta = np.array([
        [2, -2],
        [0,  6],
        ])
    Theta = cec.symmetrize_matrix(Theta)
    return Theta



def load_Theta_fcc_full():
    Theta = np.array([
        [4,  -2,  -2,   0,     0,   0,   0,   0,     0,   0],
        [0,   6,  -2,	0,    -2,   0,   0,   0,     0,   0],
        [0,   0,  18,  -4,    -6,  -2,  -2,   0,     0,   0],
        [0,   0,   0,  12,    -4,   0,  -4,   0,     0,   0],

        [0,   0,   0,   0,    30,   0,  -6,  -4,    -6,  -2],
        [0,   0,   0,   0,     0,  12,  -8,   0,     0,   0],
        [0,   0,   0,   0,     0,   0,  56,   0,   -16,  -8],
        [0,   0,   0,   0,     0,   0,   0,  12,    -4,   0],

        [0,   0,   0,   0,     0,   0,   0,   0,    66, -14],
        [0,   0,   0,   0,     0,   0,   0,   0,     0,  48],
        ])
    Theta = cec.symmetrize_matrix(Theta)
    return Theta






def calc_sigma_dUss(self, wc, zetac, t='fcc_partial'):
    b = self.b 
    cn = self.cn 
    nelem = self.nelem 
    EPI = self.EPI

    if t=='fcc_partial':
        Theta = load_Theta_fcc_partial()
        tk = 2

    elif t=='fcc_full':
        Theta = load_Theta_fcc_full()
        tk = 1

    dmax = np.min([ EPI.shape[0], Theta.shape[0] ])


    print('==> calculating s2:')
    s2 = 0 
    for n1 in np.arange(nelem):
        for n2 in np.arange(nelem):
            for d1 in np.arange(dmax):
                for d2 in np.arange(dmax):
                    s2 = s2 +1/4 * cn[n1]*cn[n2] * EPI[d1, n1, n2]*EPI[d2, n1, n2] * Theta[d1, d2]
    print(s2)


    for n1 in np.arange(nelem):
        for n2 in np.arange(nelem):
            for n3 in np.arange(nelem):
                for d1 in np.arange(dmax):
                    for d2 in np.arange(dmax):
                        s2 = s2 -1/2 * cn[n1]*cn[n2]*cn[n3] * EPI[d1, n1, n2]*EPI[d2, n1, n3] * Theta[d1, d2]
    print(s2)


    for n1 in np.arange(nelem):
        for n2 in np.arange(nelem):
            for n3 in np.arange(nelem):
                for n4 in np.arange(nelem):
                    for d1 in np.arange(dmax):
                        for d2 in np.arange(dmax):
                            s2 = s2 +1/4 * cn[n1]*cn[n2]*cn[n3]*cn[n4] * EPI[d1, n1, n2]*EPI[d2, n3, n4] * Theta[d1, d2]
    print(s2)
    

    sigma_dUss = np.sqrt( tk *zetac/(np.sqrt(3)*b) *wc/(b/2) * s2)
    return sigma_dUss 
            







#==============================

def calc_std_gamma_APB(self, l1, l2, param={}):

    self.calc_b_from_V0()
    b = self.b

    sigma_dUss  = calc_sigma_dUss(self, l1, l2, t='fcc_partial')
    sigma_gamma_APB  = sigma_dUss / (l1 * l2) *self.qe*1e20*1e3

    sigma_dUss_f = calc_sigma_dUss(self, l1, l2, t='fcc_full')
    sigma_gamma_APB_f = sigma_dUss_f / (l1 * l2) *self.qe*1e20*1e3


    if 'filename' in param: 
    
        filen = 'slip_' + param['filename'] + '.txt'
        f = open(filen,"w+")
        
        f.write('# std of gamma_APB in one slip: \n' )
        f.write('%16s %16s %16s \n' \
        %('b (Ang)', 'l1/b', 'l2/b' ) )
        f.write('%16.8f %16.8f %16.8f \n\n' \
        %(b, l1/b, l2/b ) )


        f.write('# partial: \n' )
        f.write('%16s %33s \n' \
        %('sigma_dUss (eV)', 'sigma_gamma_APB (mJ^2)' ) )
        f.write('%16.8f %33.8f \n\n' \
        %(sigma_dUss, sigma_gamma_APB ) )


        f.write('# full: \n' )
        f.write('%16s %33s \n' \
        %('sigma_dUss_f (eV)', 'sigma_gamma_APB_f (mJ^2)' ) )
        f.write('%16.8f %33.8f \n\n' \
        %(sigma_dUss_f, sigma_gamma_APB_f ) )


        f.close() 

    return sigma_dUss_f 













# =====================================
# average strengthening of SRO
# =====================================




def load_ndd_fcc_full():
    ndd = np.array([
        [  0,  1,  1,    0,  0,  0,    0 ],
        [  0,  0,  1,    0,  1,  0,    0 ],
        [  0,  0,  0,    2,  3,  1,    1 ],
        [  0,  0,  0,    0,  2,  0,    2 ],
        [  0,  0,  0,    0,  0,  0,    3 ],
        [  0,  0,  0,    0,  0,  0,    4 ],
        [  0,  0,  0,    0,  0,  0,    0 ],
        ])
    ndd = cec.symmetrize_matrix(ndd)
    return ndd






def calc_tau_A(self, b, t='fcc_full'):
    cn = self.cn 
    nelem = self.nelem 
    EPI = self.EPI
    shellmax = self.shellmax 
    SRO = self.SRO 

    if t=='fcc_full':
        ndd = load_ndd_fcc_full()

    dmax = np.min([ SRO.shape[0], ndd.shape[0] ])


    tauA = 0
    for d1 in np.arange(shellmax):
        for n1 in np.arange(nelem):
            for n2 in np.arange(nelem):
                for d2 in np.arange(dmax):
                    tauA = tauA + cn[n1]*cn[n2] * EPI[d1, n1, n2] * ( SRO[d2, n1, n2] - SRO[d1, n1, n2] ) * ndd[d1, d2]

    
    tauA = 2*np.sqrt(2/3) /( b*np.sqrt(2) )**3 * tauA    *self.qe*1e30/1e6    #[MPa]
    return tauA 










