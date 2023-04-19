

import numpy as np 
import os 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from myvasp import vasp_func as vf 

from myvasp import vasp_epi_res 








# EPI fitting, which follows  E = X @ beta  
class class_epi_fit:
    def __init__(self, lpairs, E):
        vf.confirm_0( len(lpairs.leta) - len(E), str1='wrong len(leta_res) ' )
        self.lpairs = lpairs
        self.E = E




    #-----------------------------
    
    def routine_1(self, sd, tt=0.8, dmax=5, islip='fcc_partial', fname_suffix=''):
        ntrain = np.around( len(self.lpairs.leta) * tt )
        print('ntrain:', ntrain) 

        epi_res1 = self.calc_epi(dmax=dmax, ntrain=ntrain) 
        epi_res1.save_epi_res(fname_suffix=fname_suffix) 
        epi_res1.write_epi_res(fname_suffix=fname_suffix) 

        print('==> scan ntrain')
        self.calc_epi(dmax=dmax, ntrain=ntrain, scan_ntrain=True, fname_suffix=fname_suffix) 
        fname1 = 'lepi_res_ntrain_%s.pkl'  %(fname_suffix)
        self.calc_sdUss_tilde(filename=fname1, islip = islip) 
        self.plot_lepi_res_ntrain(filename=fname1, sd=sd) 
        if fname_suffix == 'ssf':
            self.plot_lepi_res_ntrain_2(filename=fname1) 

        print('==> scan dmax')
        dmax_max = np.min([ self.lpairs.shellmax, 11 ])
        self.scan_dmax(dmax_range=np.arange(2, dmax_max), ntrain=ntrain, fname_suffix=fname_suffix) 
        fname2 = 'lepi_res_dmax_%s.pkl'  %(fname_suffix)
        self.calc_sdUss_tilde(filename=fname2, islip = islip) 
        self.plot_lepi_res_dmax(filename=fname2) 




    #=============================

    def calc_epi(self, dmax, ntrain, scan_ntrain=False, fname_suffix=''): 
        X = self.lpairs.get_X(dmax) 
        E = self.E.copy()  

        if scan_ntrain == False:
            ntrain_range = np.array([ntrain]) 
        else:
            temp = ( (self.lpairs.nelem+1) * self.lpairs.nelem -2) /2 * dmax 
            ntrain_min = int( np.ceil( (temp+2)/10 )*10 )
            ntrain_range = np.arange(ntrain_min, ntrain+1) 

        lepi_res = [] 
        for i in ntrain_range:
            X_train, X_test = vf.split_train_test(X, i) 
            E_train, E_test = vf.split_train_test(E, i) 

            lstsq_res = np.linalg.lstsq(X_train, E_train, rcond=None)       
            beta = lstsq_res[0]           
            E_p  = X @ beta    # EPI predicted energy            

            epi_res1 = vasp_epi_res.class_epi_res( self.lpairs.nelem, self.lpairs.r_shell, dmax, i, lstsq_res, X, E, E_p)
            lepi_res.append( epi_res1 )
            
        if scan_ntrain == False:
            return lepi_res[0]
        else:
            if fname_suffix != '':
                filename = 'lepi_res_ntrain_%s.pkl'  %(fname_suffix) 
            else:
                filename = 'lepi_res_ntrain.pkl'    
            vf.my_save_pkl(lepi_res, filename)




    #=============================

    def scan_dmax(self, dmax_range, ntrain, fname_suffix=''):
        lepi_res = [] 
        for dmax in dmax_range:
            epi_res1 = self.calc_epi(dmax, ntrain) 
            lepi_res.append( epi_res1 )

        if fname_suffix != '':
            filename = 'lepi_res_dmax_%s.pkl'  %(fname_suffix) 
        else:
            filename = 'lepi_res_dmax.pkl'  
        vf.my_save_pkl(lepi_res, filename)




    def calc_sdUss_tilde(self, filename, cn=np.array([1, 1]), islip='fcc_partial' ):
        from myalloy import main as myalloy_main
        a1 = myalloy_main.alloy_class(name='fcc_alloy', cn=cn, brav_latt='fcc')
        print('islip:', islip)

        lepi_res = vf.my_read_pkl(filename) 
        sigma_dUss_tilde = np.array([]) 

        for i in np.arange( len(lepi_res) ):
            a1.set_EPI( lepi_res[i].epi )
            temp = a1.calc_sigma_dUss_tilde(t = islip)  
            sigma_dUss_tilde = np.append( sigma_dUss_tilde, temp )      

        filename2 = '%s.sdUss_tilde.txt' %(filename[0:-4])
        np.savetxt(filename2, sigma_dUss_tilde)




    #============================================
    # plot lepi_res - ntrain 
    #============================================

    def plot_lepi_res_ntrain(self, filename, sd):      
        lepi_res = vf.my_read_pkl(filename) 
        
        Ntot = np.prod(sd)*6
        Ntot_m = Ntot / (sd[0] * sd[1])
        Ntot_s = Ntot / np.sqrt(sd[0]*2 * sd[1])
 
        dmax   = np.array([])
        ntrain = np.array([])

        mE_train = np.array([])
        sE_train = np.array([])

        mE_p_train = np.array([])
        sE_p_train = np.array([])

        Vnm = np.zeros(lepi_res[0].dmax)   # a specific pair nm 

        for i in np.arange( len(lepi_res) ):
            dmax   = np.append( dmax,   lepi_res[i].dmax   )  
            ntrain = np.append( ntrain, lepi_res[i].ntrain )  

            mE_train   = np.append( mE_train,   lepi_res[i].E_train.mean()   )    
            sE_train   = np.append( sE_train,   lepi_res[i].E_train.std()    )    

            mE_p_train = np.append( mE_p_train, lepi_res[i].E_p_train.mean() )    
            sE_p_train = np.append( sE_p_train, lepi_res[i].E_p_train.std()  )
       
            Vnm = np.vstack([Vnm, lepi_res[i].epi[:,0,1]])
        Vnm=np.delete(Vnm, 0, 0) 


        ax1 = create_ax1() 
        fig_xlim = [0, ntrain[-1]]

        for j in np.arange(Vnm.shape[1]):
            str1 = '$d_{%d}$'  %(j+1)
            ax1[0].plot( ntrain, Vnm[:,j], '-', label=str1 )
        ax1[0].plot( fig_xlim, [0, 0], '--', color='gray' ) 


        ax1[1].plot( ntrain, mE_train   *Ntot_m, '-', color='k',  label='from atomistically computed energies ')
        ax1[1].plot( ntrain, mE_p_train *Ntot_m, '-', color='C3', label='from EPI-predicted energies')
        # ax1[1].plot( fig_xlim, [0, 0], '--', color='gray' ) 


        ax1[2].plot( ntrain, sE_train   *Ntot_s, '-', color='k',  label='from atomistically computed energies ')
        ax1[2].plot( ntrain, sE_p_train *Ntot_s, '-', color='C3', label='from EPI-predicted energies')

        filename2 = '%s.sdUss_tilde.txt' %(filename[0:-4])
        if os.path.exists(filename2) :
            sigma_dUss_tilde = np.loadtxt(filename2)  
            ax1[2].plot( ntrain, sigma_dUss_tilde, '-',  color='C2', label='analytical $\\widetilde{\\sigma}_{\\Delta U_{s-s}}$')

      
        ax1[0].legend( fontsize=6, loc='lower left', ncol=2)       
        ax1[1].legend( fontsize=6, loc='lower left' )       
        ax1[2].legend( fontsize=6, loc='lower left' )       

        ax1[2].set_xlim( fig_xlim )
        ax1[2].set_xlabel('$N_\\mathrm{train}$')

        ax1[0].set_ylim([-0.06, 0.04]) 
        # ax1[1].set_ylim([-0.02, 0.02])      
        # ax1[2].set_ylim([0, 0.06]) 

        ax1[0].set_ylabel('EPI $V_{\\mathrm{AuNi}, d}$ (eV)')
        ax1[1].set_ylabel('mean of $ E_\\mathrm{tot} /(N_1 N_2) $ (eV)')
        ax1[2].set_ylabel( 'std of $ E_\\mathrm{tot} /\\sqrt{2 N_1 N_2} $ (eV)')         
        

        vf.confirm_0( dmax.std() )
        str1 = 'EPI $N_d = %d$' %( dmax[0] ) 
        vf.my_text(ax1[0], str1, 0.5, 0.9, ha='center' )

        create_abc(ax1)

        figname = '%s.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')




    def plot_lepi_res_ntrain_2(self, filename):      
        lepi_res = vf.my_read_pkl(filename) 
        
        dmax   = np.array([])
        ntrain = np.array([])
        
        Vnm = np.zeros(lepi_res[0].dmax)   # a specific pair nm 
        
        for i in np.arange( len(lepi_res) ):
            dmax   = np.append( dmax,   lepi_res[i].dmax   )  
            ntrain = np.append( ntrain, lepi_res[i].ntrain )  
            
            Vnm = np.vstack([Vnm, lepi_res[i].epi[:,0,1]])
        Vnm=np.delete(Vnm, 0, 0) 
        
        
        ax1 = create_ax2() 
        fig_xlim = [0, ntrain[-1]]
        print(fig_xlim) 

        for j in np.arange(Vnm.shape[1]):
            str1 = '$d_{%d}$'  %(j+1)
            ax1.plot( ntrain, Vnm[:,j], '-', label=str1 )        
        ax1.plot( fig_xlim, [0, 0], '--', color='gray' ) 
        

        ax1.legend( fontsize=6, loc='lower left', ncol=2)            
        
        ax1.set_xlim( fig_xlim )
        ax1.set_xlabel('$N_\\mathrm{train}$')
        
        ax1.set_ylim([-0.06, 0.04]) 
        ax1.set_ylabel('EPI $V_{\\mathrm{AuNi}, d}$ (eV)')           
        

        vf.confirm_0( dmax.std() )          
        str1 = 'EPI $N_d = %d$' %( dmax[0] ) 
        vf.my_text(ax1, str1, 0.5, 0.9, ha='center' )
        
        figname = '%s_2.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')






    #============================================
    # plot lepi_res - dmax 
    #============================================

    def plot_lepi_res_dmax(self, filename):      
        lepi_res = vf.my_read_pkl(filename) 

        dmax     = np.array([])
        ntrain   = np.array([])
        pe_train = np.array([])
        pe_test  = np.array([])
        sE_train = np.array([])
        sE_test  = np.array([])

        Vnm = np.ones([ len(lepi_res), lepi_res[-1].dmax  ]) *1e6

        for i in np.arange( len(lepi_res) ):
            dmax     = np.append( dmax,     lepi_res[i].dmax      )
            ntrain   = np.append( ntrain,   lepi_res[i].ntrain    )  
            pe_train = np.append( pe_train, lepi_res[i].pe_train  )    
            pe_test  = np.append( pe_test,  lepi_res[i].pe_test   )    
            sE_train = np.append( sE_train, lepi_res[i].E_train.std() )    
            sE_test  = np.append( sE_test,  lepi_res[i].E_test.std()  )    

            Vnm[i, 0:(lepi_res[i].dmax) ] = lepi_res[i].epi[:, 0, 1]  

        vf.confirm_0( ntrain.std()   )
        vf.confirm_0( sE_train.std() )
        vf.confirm_0( sE_test.std()  )


        ax1 = create_ax1() 
        fig_xlim = [dmax[0], dmax[-1]]

        for j in np.arange(Vnm.shape[1]):           
            temp = Vnm[:,j]
            mask = ( temp != 1e6 ) 
            xi = dmax[mask]
            yi = temp[mask] 
            str1 = '$d_{%d}$'  %(j+1)

            if j < 1.1:
                alpha = 1 
            elif j < 4.1:
                alpha = 0.7
            else:
                alpha = 0.4 

            ax1[0].plot( xi, yi, '-o', label=str1, alpha=alpha)  
        ax1[0].plot( fig_xlim, [0, 0], '--', color='gray' ) 


        str1 = 'training set, std = %.3f meV/atom'  %(sE_train[0]*1e3) 
        str2 =  'testing set, std = %.3f meV/atom'  %(sE_test[0] *1e3) 
        ax1[1].plot( dmax, pe_train, '-s', color='C0',  label=str1)
        ax1[1].plot( dmax, pe_test,  '-o', color='C1',  label=str2)
      
        filename2 = '%s.sdUss_tilde.txt' %(filename[0:-4])
        if os.path.exists(filename2):
            sigma_dUss_tilde = np.loadtxt(filename2)  
            ax1[2].plot( dmax, sigma_dUss_tilde, '-s', color='C2')


        ax1[0].legend( fontsize=6, loc='lower left', ncol=2)       
        ax1[1].legend( fontsize=6, loc='lower left' )       

        ax1[2].set_xlim( fig_xlim )

        # xt  = ax1[2].get_xticks()
        # xtl = ax1[2].get_xticklabels()
        # for i in np.arange(len(xt)):
        #     str1 = '$ d_{%d} $'  %( xt[i] ) 
        #     xtl[i] = str1 
        # ax1[2].set_xticks(xt) 
        # ax1[2].set_xticklabels(xtl) 

        ax1[2].set_xlabel('$N_d$')

        ax1[0].set_ylim([-0.06, 0.04]) 
        ax1[1].set_ylim([0, 1]) 
        ax1[2].set_ylim([0, 0.06]) 

        ax1[0].set_ylabel('EPI $V_{\\mathrm{AuNi}, d}$ (eV)')
        ax1[1].set_ylabel('percent error')
        ax1[2].set_ylabel('analytical $ \\widetilde{\\sigma}_{\\Delta U_{s-s}} $ (eV)')


        str1 = 'EPI $N_\\mathrm{train} = %d$' %( ntrain[0] ) 
        vf.my_text(ax1[0], str1, 0.5, 0.9, ha='center' )

        create_abc(ax1)

        figname = '%s.pdf' %(filename[0:-4])
        plt.savefig(figname)
        plt.close('all')








def create_ax1():
    fig_wh = [3, 6]
    fig_subp = [3, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
        
    fig_pos  = np.array([0.24, 0.71, 0.68, 0.26])
    for i in np.arange(fig_subp[0]):
        ax1[i].set_position( fig_pos + np.array([0, -0.32*i, 0,  0]) )

    return ax1 



def create_ax2():
    fig_wh = [3, 2.2]
    fig_subp = [1, 1]
    fig1, ax1 = vf.my_plot(fig_wh, fig_subp)
        
    fig_pos  = np.array([0.24, 0.18, 0.68, 0.75])
    ax1.set_position( fig_pos )

    return ax1 




def create_abc(ax1):
    x = -0.31
    y =  1.02

    str1 = '(a)'
    vf.my_text(ax1[0], str1, x, y, weight='bold')

    str1 = '(b)'
    vf.my_text(ax1[1], str1, x, y, weight='bold')

    str1 = '(c)'
    vf.my_text(ax1[2], str1, x, y, weight='bold')  









