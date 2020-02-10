#!/usr/bin/env python3

import numpy as np






def run_linear_reg_for_misfit(data1, data2, cn):

    distri=0

    y, X = create_input(data1, data2, distri)
    beta, R2 = linear_reg(y, X)
    misfit = cal_misfit(beta, cn)
    
    write_output(data1, data2, cn, beta, R2, misfit)
   
   





def run_linear_reg_for_precision_uncertainty(data1, data2, cn):

    nrand=100000
   
    nsamples=data1.shape[0]
    nelem   =data1.shape[1]



    distri=0

    y, X = create_input(data1, data2, distri)
    beta, R2 = linear_reg(y, X)
    misfit = cal_misfit(beta, cn)
    sigmay = cal_sigmay(misfit)


    ytot=np.array([y])
    Xtot=np.array([X])
    mtot=np.array([misfit])
    stot=np.array([sigmay])
    

    distri=1
    
    for i in range(0, nrand-1, 1):
        y, X = create_input(data1, data2, distri)
        beta, R2 = linear_reg(y, X)
        misfit = cal_misfit(beta, cn)
        sigmay = cal_sigmay(misfit)
    
        ytot=np.concatenate([ytot, y[None,...]], axis=0)
        Xtot=np.concatenate([Xtot, X[None,...]], axis=0)
        mtot=np.concatenate([mtot, misfit[None,...]], axis=0)
        stot=np.concatenate([stot, sigmay[None,...]], axis=0)
    
    print(ytot.shape, Xtot.shape, mtot.shape, stot.shape)



    f = open("misfit_precision_uncertainty.txt","w+")
    
    f.write("# inputs: c_1 to c_N-1, a\n")
    
    for j1 in range(0, nsamples, 1):
        for j2 in range(1, nelem, 1):
    
            temp=np.array([])
            for i in range(0, nrand, 1):
                temp=np.append(temp, Xtot[i][j1][j2])
    
            f.write("%12.8f %12.8f \n"  %(temp.mean(), temp.std()) )
    
    
        temp=np.array([])
        for i in range(0, nrand, 1):
            temp=np.append(temp, v2a([ ytot[i][j1] ]) )
    
        f.write("%12.8f %12.8f \n"  %(temp.mean(), temp.std()) )
        f.write(" \n")
    
    
    # check a0

    temp=np.array([])
    for i in range(0, nrand, 1):
        temp=np.append(temp, v2a([ mtot[i][0] ]) )

    print(temp.shape)

    f.write("# a0 \n")
    f.write("%12.8f %12.8f %12.8f \n\n"  \
    %(temp.mean(), temp.std(),  temp.std()/temp.mean()  ) )

 
    # check delta
    
    temp=np.array([])
    for i in range(0, nrand, 1):
        temp=np.append(temp, mtot[i][1] )
    
    f.write("# delta \n")
    f.write("%12.8f %12.8f %12.8f \n\n"  \
    %(temp.mean(), temp.std(),  temp.std()/temp.mean()  ) )
    
    
   
    # check sigmay
    
    temp=np.array([])
    for i in range(0, nrand, 1):
        temp=np.append(temp, stot[i] )
    
    f.write("# sigmay \n")
    f.write("%12.8f %12.8f %12.8f \n\n"  \
    %(temp.mean(), temp.std(),  temp.std()/temp.mean()  ) )
    
 
    f.close()









#=====================
# functions
#=====================



def a2v(a):
    V = a**3/4
    return V





def v2a(V):
    a=np.array([])
    for i in range(0, len(V), 1):
        a=np.append(a, (V[i]*4)**(1/3) )
    return a







def create_input(data1, data2, distri):
    y=np.array([])
    X=np.array([])

    nsamples=data1.shape[0]
    nelem=data1.shape[1]

#    print("nsamples, nelem:")
#    print(nsamples, nelem)
   
    data12 = np.copy( data1 )
 

    if distri==1:
        for i in range(0, nsamples, 1):
            for j in range(0, nelem, 1):
                data12[i][j] = np.random.normal( data1[i][j], data2[i][j], 1) 
    
    y = np.copy( data12[0:, -1] )

    temp1 = np.ones( (nsamples, 1) )
    temp2 = np.copy( data12[0:, 0:-1]  )
    X = np.hstack( (temp1, temp2) )

    return y, X
 
       





def linear_reg(y, X):
    a= X.transpose() @ X
    b= X.transpose() @ y
   
    beta=np.linalg.solve( a, b )


    f=X@beta

    SSres=np.sum( np.square(y-f) )
    SStot=np.sum( np.square(y- np.mean(y)) )
   
    R2= 1-SSres/SStot

    return beta, R2




def cal_misfit(beta, cn):

    nelem=len(beta)
    
    qX=np.hstack( (1, cn) )
   
    # atomic volume at cn
    V0= qX @ beta

    
    # derivatives
    k=np.hstack( (beta[1:], 0))
    
    cn=np.append(cn, 1-np.sum(cn))

    cnk= cn @ k

    dV= np.array([])
    for i in range(0, nelem, 1):
        dV = np.append( dV, (k[i] -cnk) )

    delta= np.sqrt( cn @ np.square(dV) ) /V0/3
   
    misfit = np.append(V0, delta)
    misfit = np.append(misfit, dV) 
    return misfit





def write_output(data1, data2, cn, beta, R2, misfit):
    
    nsamples=data1.shape[0]
    nelem   =data1.shape[1]

    cn=np.append(cn, 1-np.sum(cn))
   
    
    f = open("misfit_results.txt","w")


    f.write("%12s %12s %12s \n" %('V0', 'delta', 'a0_fcc' ) )
    f.write("%12.8f %12.8f %12.8f \n\n" \
    %(misfit[0], misfit[1], v2a([misfit[0]]) ) )

    f.write("# dV: \n" )
    for j in range(0, nelem, 1):
        f.write("%12.8f " %( misfit[j+2] ) )
    f.write("\n\n" )


    f.write("# V0+dV: \n" )
    for j in range(0, nelem, 1):
        f.write("%12.8f " %( misfit[j+2]+misfit[0] ) )
    f.write("\n\n\n" )





    f.write("# beta: \n" )
    for j in range(0, nelem, 1):
        f.write("%12.8f " %( beta[j] ) )
    f.write("\n\n" )

    f.write("%12s \n" %('R2' ) )
    f.write("%12.8f \n\n\n" %(R2) )



    f.write("%12s %12s \n" %('nsamples', 'nelem' ) )
    f.write("%12d %12d \n\n" %(nsamples, nelem ) )

    f.write("# data1, c_N, a : \n" )
    for i in range(0, nsamples, 1):
        for j in range(0, nelem, 1):
            f.write("%12.8f " %( data1[i][j] ) )

        f.write("  %12.8f " %( 1-np.sum(data1[i, 0:-1]) ) )

        f.write("%12.8f " %( v2a([ data1[i][-1] ]) ) )

        f.write("\n" )
    f.write("\n" )

    f.write("# data2: \n" )
    for i in range(0, nsamples, 1):
        for j in range(0, nelem, 1):
            f.write("%12.8f " %( data2[i][j] ) )
        f.write("\n" )
    f.write("\n" )

    f.write("# cn: \n" )
    for j in range(0, nelem, 1):
        f.write("%12.8f " %( cn[j] ) )
    f.write("\n\n" )











def cal_sigmay(misfit):
    
    kB=1.38064852e-23
    alpha=1/8
    et0=1e4
    
    T=300
    et=1e-3
    
    b=v2a( [misfit[0]] ) /np.sqrt(2)
    delta=misfit[1]

    mu111 = 78.4
    mu = 103.6
    nu = 0.2708
 

    Gamma = alpha * mu111 * b**2
    P= mu*(1+nu)/(1-nu)

    dEb = 2.5785 * (Gamma/b**2)**( 1/3) * b**3 * P**(2/3) * delta**(2/3)
    ty0 = 0.04865* (Gamma/b**2)**(-1/3)        * P**(4/3) * delta**(4/3)

    dEb=dEb * 1e9 * 1e-30   # [J]

    ty  = ty0 *( 1- ( kB*T/dEb * np.log(et0/et) )**(2/3) )
    sigmay = 3.06 * ty

    return sigmay 






 
