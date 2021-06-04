
import numpy as np
import sys 


def stroh_slip_system(self, slip_system='basal_a_edge', bp=None):

    a=self.a
    c=self.c
    brav_latt=self.brav_latt
    
        
    if brav_latt == 'fcc':
        if slip_system == '111_a_edge' :

            # the new axes in pristine coordinates
            mm = np.array([ [1.0, 0.0, -1.0], 
                            [1.0, -1., 1.], 
                            [-1., -2., -1.]  ])

            # total b in new axes
            bt = np.array([[a/np.sqrt(2)], [0], [0]])
            
            # angle between slip plane and x1-x3 plane
            theta = 0.

            # in slip plane, xx=a1*e1 ; yy=a2*e2
            xx = bt
            yy=np.array([[0.],[0.],[-a/np.sqrt(2)*np.sqrt(3)/2]])

            if bp == None:
                bp=np.array([ 1/2, 1/3 ])
            


            
            

    if brav_latt == 'hcp':
        if slip_system == 'pyr1_ca_screw' :
            mm = np.array([ [np.sqrt(3.0),                  1.,    0.], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a, a*a/c], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a,    -c]   ])

            bt = np.array([[0.], [0.], [-np.sqrt(a**2+c**2)]])
           
            t1 = np.cross(xx.T,bt.T)
            t2 = np.array([[0.],[1.],[0.]])
            theta = np.arccos(np.dot(t1,t2)/np.dot(np.linalg.norm(t1,2),np.linalg.norm(t2,2)))
          
            xx = np.array([[(1.0/2.0)*a*np.sqrt(3.0)], [(1.0/2.0)*a*c/np.sqrt(a**2+c**2)], [(1.0/2.0)*a**2/np.sqrt(a**2+c**2)]])
            yy = bt+1.0/2.0*xx
            
            if bp == None:
                bp=np.array([ 0, 0.40 ])
            
            



        elif slip_system == 'pyr2_ca_screw' :
            mm = np.array([ [np.sqrt(3.0),                  1.,      0.], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a, a*(a/c)], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a,      -c]  ])

            bt = np.array([[0.], [0.], [-np.sqrt(a**2+c**2)]])
            theta = 0.
            xx = np.array([[a*np.sqrt(3.0)], [0.], [0.] ])
            yy = bt
    
            if bp == None:
                bp=np.array([ 0, 0.48 ])
            



    
    b1 = bp[0]* xx + bp[1]* yy
    b2 = bt - b1
    
    for i in np.arange(3):
        mm[i,:] = mm[i,:]/np.linalg.norm(mm[i,:])

    if np.dot(mm[0,:], (np.cross(mm[1,:], mm[2,:])) ) < 0 :
        sys.exit('ABORT: wrong mm order.')


    return mm, bt, theta, xx, yy, b1, b2
    





