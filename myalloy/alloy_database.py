
import numpy as np


def a2v(a):
    v=a**3/4
    return v



def fcc_elem_poly():
    # Velem, mu, nu
    data1=np.array([
        [a2v(3.803),  150.4,  0.26  ],  # Rh
        [a2v(3.839),    210,  0.26  ],  # Ir
    
        [a2v(3.891),     44,  0.39  ],  # Pd
        [a2v(3.924),     61,  0.38  ],  # Pt
       
        [a2v(3.524),     76,  0.31  ],  # Ni 
        [a2v(3.615),     48,  0.34  ],  # Cu
    
        [a2v(4.085),     30,  0.37  ],  # Ag
        [a2v(4.078),     27,  0.44  ],  # Au
        ])
    return data1


def fcc_elem_Cij():
    # Velem, Cijelem
    data1=np.array([
        [a2v(3.824),  415,  181,  186  ],  # Rh-DFT
        [a2v(3.872),  587,  231,  257  ],  # Ir
        [a2v(3.518),  276,  154,  125  ],  # Ni

        [a2v(3.943),  197,  150,   71  ],  # Pd
        [a2v(3.967),  302,  224,   58  ],  # Pt
        [a2v(3.635),  179,  119,   88  ],  # Cu

        [10.901,  296.6,  171.9,  144.0  ],  # Co (Shang et al. 2010)
        [13.941,  479.1,  223.7,  242.5  ],  # Ru (Shang et al. 2010)
        ])
    return data1


