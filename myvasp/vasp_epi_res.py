

import numpy as np 
from myvasp import vasp_func as vf 





class epi_res:
    def __init__(self, epi_type, ntrain, shellmax, beta, R2, X, E, E_p):
        self.epi_type = epi_type
        self.ntrain   = ntrain 
        self.shellmax = shellmax         

        self.beta = beta
        self.R2   = R2
        
        self.X    = X
        self.E    = E 
        self.E_p  = E_p          
        
        self.auto_add()




    def auto_add(self):
        self.ntest = self.X.shape[0] - self.ntrain   
    
        E   = ( self.E   - self.beta[0] )*1e3
        E_p = ( self.E_p - self.beta[0] )*1e3
      
        self.E_train,   self.E_test   = vf.split_train_test(E,   self.ntrain) 
        self.E_train_p, self.E_test_p = vf.split_train_test(E_p, self.ntrain) 

        self.rmse_train = vf.calc_RMSE( self.E_train, self.E_train_p )
        self.rmse_test  = vf.calc_RMSE( self.E_test,  self.E_test_p  )

        self.pe_train = self.rmse_train / self.E_train.std()      # percent error 
        self.pe_test  = self.rmse_test  / self.E_test.std() 




    def plot_epi_res(self, fname_suffix=''): 
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig_wh = [7, 2.2]
        fig_subp = [1, 2]
        fig1, ax1 = vf.my_plot(fig_wh, fig_subp, fig_sharex=False)

        fig_pos  = np.array([0.1, 0.22, 0.23, 0])
        fig_pos[-1] =  fig_pos[-2]*fig_wh[0]/fig_wh[1]
        for j in np.arange(fig_subp[1]):
            ax1[j].set_position( fig_pos + np.array([0.33*j, 0, 0,  0]) )

        rfcc, nfcc = vf.crystal_shell('fcc')
        xi = rfcc[0:self.shellmax].copy()

        if self.epi_type == 'normal':
            ax1[1].set_xlabel('$ E_{f,\\mathrm{DFT} } - E_{f}^\\mathrm{rand}$ (meV/atom)')
            ax1[1].set_ylabel('$ E_{f,\\mathrm{EPI} } - E_{f}^\\mathrm{rand}$ (meV/atom)')
        elif self.epi_type == 'diff':
            ax1[1].set_xlabel('$ \\Delta ( E_{f,\\mathrm{DFT} } ) $ (meV/atom)')
            ax1[1].set_ylabel('$ \\Delta ( E_{f,\\mathrm{EPI} } ) $ (meV/atom)')


        #=================================

        ax1[0].plot(xi, self.beta[1:], '-o')
    
        ax1[0].set_xlabel('Pair distance $d/a$')
        ax1[0].set_ylabel('EPI $V_{nm,d}$ (eV)')

        if self.epi_type == 'normal':
            str1 = '$E_f^\\mathrm{rand} = %.3f$ eV/atom' %( self.beta[0] )
            vf.my_text(ax1[0], str1, 0.5, 0.9, ha='center' )  


        #=================================

        E_train   = self.E_train.copy()  
        E_test    = self.E_test.copy()  
        E_train_p = self.E_train_p.copy()  
        E_test_p  = self.E_test_p.copy()  


        ax1[1].plot( E_train, E_train_p,   '.', label='training set',  alpha=0.6 ) 
        if self.ntest > 0.9:
            ax1[1].plot( E_test, E_test_p, '.', label='testing set',   alpha=0.4 )

        ax1[1].legend(fontsize=6, loc='upper left')   

        xlim = ax1[1].get_xlim() 
        ylim = ax1[1].get_ylim() 
        zlim = np.array([ np.min([ xlim[0], ylim[0] ]),  np.max([ xlim[1], ylim[1] ]) ])

        ax1[1].plot( zlim, zlim, '-k', alpha=0.3)
        ax1[1].set_xlim(zlim) 
        ax1[1].set_ylim(zlim) 


        #=================================

        str3 = '# of structures: train=%d, test=%d\n\nE_train = $%.3f \pm %.3f$\nE_train_p = $%.3f \pm %.3f$\nRMSE = %.3f, pe = %.3f' \
            %( len(E_train), len(E_test),    \
                np.mean(E_train), np.std(E_train),   np.mean(E_train_p), np.std(E_train_p), \
                self.rmse_train, self.pe_train  )
        vf.my_text(ax1[1], str3, 1.08, 0.6 )

        if self.ntest > 0.9:
            str3 = 'E_test = $%.3f \pm %.3f$\nE_test_p = $%.3f \pm %.3f$\nRMSE = %.3f, pe = %.3f' \
                %( np.mean(E_test), np.std(E_test),   np.mean(E_test_p), np.std(E_test_p), \
                    self.rmse_test, self.pe_test  )
            vf.my_text(ax1[1], str3, 1.08, -0.2 )

  
        # fitting error
        e_train = E_train - E_train_p  
        temp = np.cov( np.vstack([E_train_p, e_train]),  bias=True)

        vf.confirm_0( (temp[0,0]    - np.std(E_train_p)**2 )/10 , str1='cov 0'   )
        vf.confirm_0( (temp[1,1]    - np.std(e_train)**2   )/10 , str1='cov 1'   )
        vf.confirm_0( (np.sum(temp) - np.std(E_train)**2   )/10 , str1='cov sum' )

        str3 = 'fitting error:\ne_train = $%.3f \pm %.3f$\ncov = %.3f' \
            %( np.mean(e_train), np.std(e_train),       temp[0,1]  )                    
        vf.my_text(ax1[1], str3, 1.08, 0.2 )

        if fname_suffix != '':
            figname = 'epi_res_%s.pdf' %(fname_suffix)
        else:
            figname = 'epi_res.pdf'  
        
        plt.savefig(figname) 
        plt.close('all')

        self.save_epi_res(figname)
        self.write_epi_res(figname)
        



    def save_epi_res(self, figname):
        from myvasp import vasp_func as vf         
        filename = '%s.pkl' %(figname[0:-4])
        vf.my_save_pkl(self, filename)  




    def write_epi_res(self, figname):
        filename = '%s.txt' %(figname[0:-4])
        f = open(filename, "w+")

        f.write('# epi_res - EPI results: \n' ) 
        f.write('# epi_type: %s \n\n'  %(self.epi_type) )

        f.write('%16s %16s %16s %16s \n' \
            %('ntrain', 'ntest', 'shellmax', 'R2' ) )
        f.write('%16d %16d %16d %16.8f \n\n' \
            %(self.ntrain, self.ntest, self.shellmax, self.R2) )

        with np.printoptions(linewidth=200, \
            precision=8, suppress=True):

            f.write('beta = E_rand (eV), EPI (eV) \n')
            f.write(str( self.beta )+'\n\n')

            f.write('X \n')
            f.write(str( self.X )+'\n\n')

            temp = np.hstack([ self.E[:, np.newaxis], self.E_p[:, np.newaxis] ])
            f.write('E (eV), E_p (eV) \n')
            f.write(str( temp )+'\n\n')

        f.close() 
                





