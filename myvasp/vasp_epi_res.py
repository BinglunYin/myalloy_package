

import numpy as np 
from myvasp import vasp_func as vf 




class class_epi_res:
    def __init__(self, nelem, r_shell, dmax, ntrain, lstsq_res, X, E, E_p):
        self.nelem   = nelem 
        self.r_shell = r_shell     
        self.dmax    = dmax          
        self.ntrain  = ntrain 

        self.lstsq_res = lstsq_res
        
        self.X    = X
        self.E    = E 
        self.E_p  = E_p          
        
        self.auto_add()




    def auto_add(self):
        self.ntest = self.X.shape[0] - self.ntrain      
      
        self.E_train,   self.E_test   = vf.split_train_test(self.E,   self.ntrain) 
        self.E_p_train, self.E_p_test = vf.split_train_test(self.E_p, self.ntrain) 

        self.rmse_train = vf.calc_RMSE( self.E_train, self.E_p_train )
        self.rmse_test  = vf.calc_RMSE( self.E_test,  self.E_p_test  )

        self.pe_train = self.rmse_train / self.E_train.std()      # percent error 
        self.pe_test  = self.rmse_test  / self.E_test.std() 


        self.rank = self.lstsq_res[2]

        epi_rank = int( self.nelem*(self.nelem-1)/2 * self.dmax ) 

        eta_X = self.X[:, 0:epi_rank].copy()  
        temp1 = np.linalg.matrix_rank( eta_X )
        vf.confirm_0( temp1 - epi_rank )
        self.epi_rank = epi_rank 

        temp = self.X[:, epi_rank:].copy()  
        self.other_rank = np.linalg.matrix_rank( temp )
        vf.confirm_0( self.rank - self.epi_rank - self.other_rank, str1='wrong rank.' )

        beta = self.lstsq_res[0]
        vf.confirm_0( len(beta) - self.epi_rank -1 -(self.nelem-1)*self.dmax )

        self.epi = self.reform_epi_1_to_3( beta[0:epi_rank] )  
        self.U1t = beta[epi_rank]  
        self.dU  = self.reform_dU_1_to_2( beta[epi_rank+1:] )





    def reform_epi_1_to_3(self, epi):
        dmax = self.dmax 
        nelem = self.nelem      
        
        epi3 = np.zeros([dmax, nelem, nelem])
        m = -1
        for d in np.arange(dmax):
            for i in np.arange(nelem-1):
                for j in np.arange(i+1, nelem):
                    m = m+1  
                    epi3[d, i, j] = epi[m]  
                    epi3[d, j, i] = epi[m]  
        vf.confirm_0( m+1 - len(epi) )
        return epi3 




    def reform_dU_1_to_2(self, dU):
        dmax = self.dmax 
        nelem = self.nelem      
        
        dU2 = np.zeros([dmax, nelem])
        m = -1
        for d in np.arange(dmax):
            for j in np.arange(1, nelem):
                    m = m+1  
                    dU2[d, j] = dU[m]  
        vf.confirm_0( m+1 - len(dU) )
        return dU2 




    def save_epi_res(self, fname_suffix=''):
        if fname_suffix != '':
            filename = 'epi_res_%s.pkl'  %(fname_suffix) 
        else:
            filename = 'epi_res.pkl'   
        vf.my_save_pkl(self, filename)  




    def write_epi_res(self, fname_suffix=''):

        if fname_suffix != '':
            filename = 'epi_res_%s.txt'  %(fname_suffix) 
        else:
            filename = 'epi_res.txt'   

        f = open(filename, "w+")
        f.write('# epi_res - EPI results: \n' ) 

        f.write('%16s %16s %16s \n' \
            %('ntrain', 'ntest', 'dmax' ) )
        f.write('%16d %16d %16d \n\n' \
            %(self.ntrain, self.ntest, self.dmax ) )


        f.write('%16s %16s %16s \n' \
            %('rank', 'epi_rank', 'other_rank' ) )
        f.write('%16d %16d %16d \n\n' \
            %(self.rank, self.epi_rank, self.other_rank) )


        f.write('%16s %16s %16s %16s \n' \
            %('E_train.mean()', 'E_train.std()', 'E_test.mean()', 'E_test.std()' ) )
        f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
            %(self.E_train.mean(), self.E_train.std(), self.E_test.mean(), self.E_test.std() ) )


        f.write('%16s %16s %16s %16s \n' \
            %('rmse_train', 'rmse_test', 'pe_train', 'pe_test' ) )
        f.write('%16.8f %16.8f %16.8f %16.8f \n\n' \
            %(self.rmse_train, self.rmse_test, self.pe_train, self.pe_test ) )


        with np.printoptions(linewidth=200, \
            precision=8, suppress=True):

            f.write('beta = epi, U1t, dU (eV) \n')
            f.write('epi (eta): \n')
            f.write(str( self.epi )+'\n\n')

            f.write('U1t (constant): \n')
            f.write(str( self.U1t )+'\n\n')

            f.write('dU (epsilon): \n')
            f.write(str( self.dU )+'\n\n')

            f.write('X.shape \n')
            f.write(str( self.X.shape )+'\n\n')

            f.write('X \n')
            f.write(str( self.X )+'\n\n')

            f.write('E.shape \n')
            f.write(str( self.E.shape )+'\n\n')

            temp = np.hstack([ self.E[:, np.newaxis], self.E_p[:, np.newaxis] ])
            f.write('E (eV), E_p (eV) \n')
            f.write(str( temp )+'\n\n')

        f.close() 
                





