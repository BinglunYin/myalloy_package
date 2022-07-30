

import numpy as np 
import sys, os 
from myvasp import vasp_func as vf 
from myvasp import vasp_epi_res 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




# EPI fitting, which follows  E = X @ beta  
class epi_fit:

    def __init__(self, X, E):
        self.X = X
        self.E = E
      
        if (self.X[:,0] == 1).all():
            self.epi_type = 'normal'  
        elif (self.X[:,0] == 0).all():
            self.epi_type = 'diff' 
        else:
            sys.exit('ABORT: wrong X.')






    def routine_1(self, sd, tt=0.8, shellmax=5, fname_suffix=''):
        ntrain = np.around( self.X.shape[0] * tt )

        self.plot_binary_deta(fname_suffix=fname_suffix)

        epi_res1 = self.calc_epi(ntrain=ntrain, shellmax=shellmax) 
        epi_res1.plot_epi_res(fname_suffix=fname_suffix) 

        temp = np.ceil( (shellmax+2)/10 )*10 
        self.calc_lepi_res_ntrain(ntrain_range=np.arange(temp, ntrain+1), shellmax=shellmax, fname_suffix=fname_suffix) 
        fname1 = 'lepi_res_ntrain_%s.pkl'  %(fname_suffix)
        self.calc_nag_full_slip(filename=fname1) 
        self.plot_lepi_res_ntrain(filename=fname1, sd=sd) 
        self.plot_lepi_res_ntrain_2(filename=fname1) 

        temp = self.X.shape[1]
        self.calc_lepi_res_shellmax(ntrain=ntrain, shellmax_range=np.arange(1,temp), fname_suffix=fname_suffix) 
        fname2 = 'lepi_res_shellmax_%s.pkl'  %(fname_suffix)
        self.calc_nag_full_slip(filename=fname2) 
        self.plot_lepi_res_shellmax(filename=fname2) 
        self.plot_lepi_res_shellmax_2(filename=fname2) 


   



    #============================================


    def plot_binary_deta(self, shellmax=0, fname_suffix=''):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        X = self.X.copy()
        if shellmax == 0:
            shellmax = X.shape[1]-1

        fig_wh = [3, 2.2]
        fig_subp = [1, 1]
        fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
        
        fig_pos  = np.array([0.25, 0.2, 0.7, 0.75])
        ax1.set_position( fig_pos )

        rfcc, nfcc = vf.crystal_shell('fcc')
        xi = rfcc[0:shellmax].copy()

        # for i in np.arange(X.shape[0]):
        #     ax1.plot(xi, X[i, 1:(shellmax+1)]/(-0.5), '-', alpha=0.3)
    
        x_mean = np.mean(X[:,1:(shellmax+1)]/(-0.5), axis=0)
        x_std  = np.std( X[:,1:(shellmax+1)]/(-0.5), axis=0)

        ax1.errorbar(xi, x_mean, fmt="ko-", yerr=x_std)
        ax1.set_xlabel('Pair distance $d/a$')

        if self.epi_type == 'normal':
            ax1.set_ylabel('$           \\Delta \\eta_{nm, d}   $')
        elif self.epi_type == 'diff':
            ax1.set_ylabel('$ \\Delta ( \\Delta \\eta_{nm, d} ) $')

        if fname_suffix != '':
            figname = 'fig_deta_%s.pdf' %(fname_suffix)
        else:
            figname = 'fig_deta.pdf' 
      
        plt.savefig(figname)
        plt.close('all')




    def calc_epi(self, ntrain, shellmax):
        if self.epi_type == 'normal': 
            X = self.X[:, 0:shellmax+1].copy() 
        elif self.epi_type == 'diff': 
            X = self.X[:, 1:shellmax+1].copy()
      
        E = self.E.copy()  

        X_train, X_test = vf.split_train_test(X, ntrain) 
        E_train, E_test = vf.split_train_test(E, ntrain) 
    
        beta, R2 = vf.mylinreg(X_train, E_train)
   
        E_p  = X @ beta  # EPI predicted energy 

        if self.epi_type == 'diff': 
            beta = np.append(0, beta) 
     
        epi_res1 = vasp_epi_res.epi_res(self.epi_type, ntrain, shellmax, beta, R2, X, E, E_p)
        return epi_res1






    #============================================
    # calculate list of epi_res 
    #============================================


    def calc_lepi_res_ntrain(self, ntrain_range, shellmax, fname_suffix=''):
        lepi_res = [] 
        for ntrain in ntrain_range:
            epi_res1 = self.calc_epi(ntrain, shellmax) 
            lepi_res.append( epi_res1 )

        if fname_suffix != '':
            filename = 'lepi_res_ntrain_%s.pkl'  %(fname_suffix) 
        else:
            filename = 'lepi_res_ntrain.pkl'     %(fname_suffix) 
        vf.my_save_pkl(lepi_res, filename)
        



    def calc_lepi_res_shellmax(self, ntrain, shellmax_range, fname_suffix=''):
        lepi_res = [] 
        for shellmax in shellmax_range:
            epi_res1 = self.calc_epi(ntrain, shellmax) 
            lepi_res.append( epi_res1 )

        if fname_suffix != '':
            filename = 'lepi_res_shellmax_%s.pkl'  %(fname_suffix) 
        else:
            filename = 'lepi_res_shellmax.pkl'     %(fname_suffix) 
        vf.my_save_pkl(lepi_res, filename)




    def calc_nag_full_slip(self, filename, cn=np.array([1, 1]) ):
        from myalloy import main 
        a1 = main.alloy_class(name='fcc_alloy', cn=cn, brav_latt='fcc')

        lepi_res = vf.my_read_pkl(filename) 
        sigma_dUss_tilde = np.array([]) 

        for i in np.arange( len(lepi_res) ):
            a1.set_EPI( lepi_res[i].beta[1:] )
            temp = a1.calc_sigma_dUss_tilde(t='fcc_full')  
            sigma_dUss_tilde = np.append( sigma_dUss_tilde, temp ) 

        filename2 = '%s.full_slip.txt' %(filename[0:-4])
        np.savetxt(filename2, sigma_dUss_tilde)






    #============================================
    # plot list of epi_res - ntrain 
    #============================================


    def plot_lepi_res_ntrain(self, filename, sd):      
        lepi_res = vf.my_read_pkl(filename) 
        
        Ntot = np.prod(sd)*6
        Ntot2 = Ntot / np.sqrt(sd[0]*2 * sd[1])
 
        ntrain   = np.array([])
        shellmax = np.array([])

        mE_train = np.array([])
        sE_train = np.array([])

        mE_train_p = np.array([])
        sE_train_p = np.array([])

        beta = np.zeros(lepi_res[0].beta.shape)

        for i in np.arange( len(lepi_res) ):
            ntrain     = np.append( ntrain,     lepi_res[i].ntrain           )  
            shellmax   = np.append( shellmax,   lepi_res[i].shellmax         )  

            mE_train   = np.append( mE_train,   lepi_res[i].E_train.mean()   )    
            sE_train   = np.append( sE_train,   lepi_res[i].E_train.std()    )    

            mE_train_p = np.append( mE_train_p, lepi_res[i].E_train_p.mean() )    
            sE_train_p = np.append( sE_train_p, lepi_res[i].E_train_p.std()  )

            beta = np.vstack([beta, lepi_res[i].beta])
        beta=np.delete(beta, 0, 0) 


        ax1 = create_ax1() 
        fig_xlim = [0, ntrain[-1]]

        ax1[0].plot( ntrain, mE_train  /1e3 *Ntot2, '-', color='k',  label='from atomistically computed energies ')
        ax1[0].plot( ntrain, mE_train_p/1e3 *Ntot2, '-', color='C3', label='from EPI-predicted energies')
        ax1[0].plot( fig_xlim, [0, 0], '--', color='gray' ) 

        ax1[1].plot( ntrain, sE_train  /1e3 *Ntot2, '-', color='k',  label='from atomistically computed energies ')
        ax1[1].plot( ntrain, sE_train_p/1e3 *Ntot2, '-', color='C3', label='from EPI-predicted energies')

        filename2 = '%s.full_slip.txt' %(filename[0:-4])
        if os.path.exists(filename2):
            sigma_dUss_tilde = np.loadtxt(filename2)  
            ax1[1].plot( ntrain, sigma_dUss_tilde, '--', color='C2', label='EPI-based theoretical $ \\widetilde{\\sigma}_{\\Delta U_{s-s}} $')

        for j in np.arange(1, beta.shape[1]):
            str1 = '$d_{%d}$'  %(j)
            ax1[2].plot( ntrain, beta[:,j], '-', label=str1 )


        ax1[0].legend( fontsize=6, loc='lower left' )       
        ax1[1].legend( fontsize=6, loc='lower left' )       
        ax1[2].legend( fontsize=6, loc='lower left', ncol=2)       

        ax1[2].set_xlim( fig_xlim )

        if self.epi_type == 'normal':
            ax1[0].set_ylabel('mean of $( \\widetilde{E}_\\mathrm{tot} - \\widetilde{E}^\\mathrm{rand}_\\mathrm{tot} )$ (eV)')
            ax1[1].set_ylabel( 'std of $( \\widetilde{E}_\\mathrm{tot} - \\widetilde{E}^\\mathrm{rand}_\\mathrm{tot} )$ (eV)')
           
        elif self.epi_type == 'diff':
            ax1[0].set_ylabel('mean of $  \\Delta( \\widetilde{E}_\\mathrm{tot} )$ (eV)')
            ax1[1].set_ylabel( 'std of $  \\Delta( \\widetilde{E}_\\mathrm{tot} )$ (eV)')

            ax1[0].set_ylim([-0.02, 0.02 ]) 
            ax1[1].set_ylim([0, 0.06]) 
            ax1[2].set_ylim([-0.06, 0.04]) 

        ax1[2].set_ylabel('EPI $V_{\\mathrm{AuNi}, d}$ (eV)')
        ax1[2].set_xlabel('$n_\\mathrm{train}$')

        vf.confirm_0( shellmax.std() )
        str1 = 'EPI $d_\\mathrm{max}=d_{%d}$' %( lepi_res[0].shellmax ) 
        vf.my_text(ax1[0], str1, 0.5, 0.9, ha='center' )

        figname = '%s.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')




    def plot_lepi_res_ntrain_2(self, filename):      
        lepi_res = vf.my_read_pkl(filename) 

        ntrain     = np.array([])
        R2         = np.array([])
        rmse_train = np.array([])
        rmse_test  = np.array([])

        for i in np.arange( len(lepi_res) ):
            ntrain     = np.append( ntrain,     lepi_res[i].ntrain     )  
            R2         = np.append( R2,         lepi_res[i].R2         )    
            rmse_train = np.append( rmse_train, lepi_res[i].rmse_train )    
            rmse_test  = np.append( rmse_test,  lepi_res[i].rmse_test  )


        ax1 = create_ax1() 
        fig_xlim = [0, ntrain[-1]]

        ax1[0].plot( ntrain, R2, '-', color='k')
        
        ax1[1].plot( ntrain, rmse_train, '-', color='C0', label='training set')
        ax1[1].plot( ntrain, rmse_test,  '-', color='C1', label='testing set')
       
        ax1[1].legend( fontsize=6 )       

        ax1[0].set_ylim([0, 1]) 
        for i in np.arange(1, 2):
            temp = ax1[i].get_ylim() 
            ax1[i].set_ylim([0, np.ceil(temp[1]*10)/10 ]) 

        ax1[2].set_xlim( fig_xlim )

        ax1[0].set_ylabel('$R^2$')
        ax1[1].set_ylabel('RMSE (meV/atom)')
        ax1[2].set_ylabel(' ')
        ax1[2].set_xlabel('$n_\\mathrm{train}$')

        str1 = 'EPI $d_\\mathrm{max}=d_{%d}$' %( lepi_res[0].shellmax ) 
        vf.my_text(ax1[0], str1, 0.5, 0.9, ha='center' )

        figname = '%s_2.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')






    #============================================
    # plot list of epi_res - shellmax 
    #============================================


    def plot_lepi_res_shellmax(self, filename):      
        lepi_res = vf.my_read_pkl(filename) 

        fig_wh = [3, 2.2]
        fig_subp = [1, 1]
        fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
        
        fig_pos  = np.array([0.25, 0.2, 0.7, 0.75])
        ax1.set_position( fig_pos )

        rfcc, nfcc = vf.crystal_shell('fcc')
        lmarker='osd^>v<'

        for i in np.arange( len(lepi_res) ):
            xi = rfcc[0: lepi_res[i].shellmax].copy()
            yi = lepi_res[i].beta[1:].copy()  

            str1 = '$d_\\mathrm{max}=d_{%d}$'  %(len(xi)) 
            ax1.plot(xi, yi, '-', marker=lmarker[ np.mod(i,7) ], alpha=0.5, label=str1)
    
        ax1.legend( ncol=2, fontsize=4)       
        ax1.set_xlabel('Pair distance $d/a$')
        ax1.set_ylabel('EPI $V_{\\mathrm{AuNi},d}$ (eV)')

        str1 = 'EPI $n_\\mathrm{train}=%d$' %( lepi_res[0].ntrain ) 
        vf.my_text(ax1, str1, 0.5, 0.9, ha='center' )

        figname = '%s.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')




    def plot_lepi_res_shellmax_2(self, filename):      
        lepi_res = vf.my_read_pkl(filename) 

        ntrain     = np.array([])
        shellmax   = np.array([])
        R2         = np.array([])
        rmse_train = np.array([])
        rmse_test  = np.array([])

        for i in np.arange( len(lepi_res) ):
            ntrain     = np.append( ntrain,     lepi_res[i].ntrain     )  
            shellmax   = np.append( shellmax,   lepi_res[i].shellmax   )
            R2         = np.append( R2,         lepi_res[i].R2         )               
            rmse_train = np.append( rmse_train, lepi_res[i].rmse_train )    
            rmse_test  = np.append( rmse_test,  lepi_res[i].rmse_test  )    

        vf.confirm_0( ntrain.std() )

        ax1 = create_ax1() 
        fig_xlim = [shellmax[0], shellmax[-1]]

        ax1[0].plot( shellmax, R2, '-o', color='k')

        ax1[1].plot( shellmax, rmse_train, '-o', color='C0',  label='training set')
        ax1[1].plot( shellmax, rmse_test,  '-s', color='C1',  label='testing set')
      
        filename2 = '%s.full_slip.txt' %(filename[0:-4])
        if os.path.exists(filename2):
            sigma_dUss_tilde = np.loadtxt(filename2)  
            ax1[2].plot( shellmax, sigma_dUss_tilde, '-o', color='C2')

        ax1[1].legend( fontsize=6 )       

        ax1[0].set_ylim([0, 1]) 
        for i in np.arange(1, 3):
            temp = ax1[i].get_ylim() 
            ax1[i].set_ylim([0, np.ceil(temp[1]*10)/10 ]) 

        ax1[2].set_xlim( fig_xlim )

        ax1[0].set_ylabel('$R^2$')
        ax1[1].set_ylabel('RMSE (meV/atom)')
        ax1[2].set_ylabel('EPI-based theoretical $ \\widetilde{\\sigma}_{\\Delta U_{s-s}} $')
        ax1[2].set_xlabel('$d_\\mathrm{max}$')

        str1 = 'EPI $n_\\mathrm{train}=%d$' %( lepi_res[0].ntrain ) 
        vf.my_text(ax1[0], str1, 0.5, 0.9, ha='center' )

        figname = '%s_2.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')






def create_ax1():
    fig_wh = [3, 6]
    fig_subp = [3, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
        
    fig_pos  = np.array([0.25, 0.71, 0.67, 0.27])
    for i in np.arange(fig_subp[0]):
        ax1[i].set_position( fig_pos + np.array([0, -0.32*i, 0,  0]) )

    return ax1 












