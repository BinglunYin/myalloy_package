
import numpy as np
import sys 


def slip_system(self, slip_system='111_a_edge', param={}):

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

            if 'bp' in param: 
                bp = param['bp']
            else:
                bp=np.array([ 1/2, 1/3 ])
            


            
            

    if brav_latt == 'hcp':
        if slip_system == 'pyr1_ca_screw' :
            mm = np.array([ [np.sqrt(3.0),                  1.,    0.], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a, a*a/c], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a,    -c]   ])

            ca = np.sqrt(a**2+c**2)

            bt = np.array([[0.], [0.], [-ca]])
           
            xx = np.array([[(1.0/2.0)*a*np.sqrt(3.0)], [(1.0/2.0)*a*c/ca], [(1.0/2.0)*a**2/ca]])
            yy = bt+1.0/2.0*xx
            
            t1 = np.cross(xx.T, bt.T)
            t2 = np.array([[0.],[1.],[0.]])
            temp = np.dot(t1, t2)
            theta = np.arccos( temp[0,0]/( np.linalg.norm(t1, 2) * np.linalg.norm(t2, 2)) )
                      
            if 'bp' in param: 
                bp = param['bp']
            else:
                bp=np.array([ 0, 0.40 ])
            
            



        elif slip_system == 'pyr2_ca_screw' :
            mm = np.array([ [np.sqrt(3.0),                  1.,      0.], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a, a*(a/c)], 
                            [       a/2.0, -np.sqrt(3.0)/2.0*a,      -c]  ])

            bt = np.array([[0.], [0.], [-np.sqrt(a**2+c**2)]])
            theta = 0.
            xx = np.array([[a*np.sqrt(3.0)], [0.], [0.] ])
            yy = bt
    
            if 'bp' in param: 
                bp = param['bp']
            else:
                bp=np.array([ 0, 0.48 ])
            






# <a> cross-slip in hcp

        elif slip_system == 'basal_a_screw' :
            mm = np.array([ [0.0, 1, 0], 
                            [  0, 0, 1], 
                            [  1, 0, 0] ])

            bt = np.array([ [0.], [0.], [a] ])
            theta = 0.
            xx = bt 
            yy = np.array([ [np.sqrt(3)/2*a], [0], [0] ])
    
            if 'bp' in param: 
                bp = param['bp']
            else:
                bp=np.array([ 0.5, 1/3 ])
            


        elif slip_system == 'prism_a_screw' :
            mm = np.array([ [0.0, 1, 0], 
                            [  0, 0, 1], 
                            [  1, 0, 0] ])

            bt = np.array([ [0.], [0.], [a] ])
            theta = np.pi/2
            xx = bt 
            yy = np.array([ [0.0], [c], [0] ])
    
            if 'bp' in param: 
                bp = param['bp']
            else:
                bp=np.array([ 0.5, 0 ])
            


        elif slip_system == 'pyr1_a_screw' :
            mm = np.array([ [0.0, 1, 0], 
                            [  0, 0, 1], 
                            [  1, 0, 0] ])

            bt = np.array([ [0.], [0.], [a] ])
            theta = np.arctan( c/(np.sqrt(3)/2*a) )
            xx = bt 
            yy = np.array([ [np.sqrt(3)/2*a], [c], [0] ])
    
            if 'bp' in param: 
                bp = param['bp']
            else:
                bp=np.array([ 0.5, -0.116 ])
            






    
    b1 = bp[0]* xx + bp[1]* yy
    b2 = bt - b1
    
    for i in np.arange(3):
        mm[i,:] = mm[i,:]/np.linalg.norm(mm[i,:])

    if np.dot(mm[0,:], (np.cross(mm[1,:], mm[2,:])) ) < 0 :
        sys.exit('ABORT: wrong mm order.')



    return mm, theta, b1, b2
    





