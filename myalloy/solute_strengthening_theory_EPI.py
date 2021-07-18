
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
        [4,  -2,  -2,   0],
        [0,   6,  -2,	0],
        [0,   0,  18,  -4],
        [0,   0,   0,  12],
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
    elif t=='fcc_full':
        Theta = load_Theta_fcc_full()

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
    

    sigma_dUss = np.sqrt( zetac/np.sqrt(3)/b * 4*wc/b * s2)
    return sigma_dUss 
            








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










