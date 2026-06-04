from pylab import *
import numpy as np
import dill, sys, os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

#from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parent.parent))
from HidKim_K2Hawkes import k2_hawkes_rfm, iclr2026_scenario


# Read Data ##############################################
# T = 2000/3000/5000/7000
# d_spk: (occurrence time, dimension label)
# k_true: true triggering kernel
# T: End time of observation
# mu: true baseline intensity

dfile = 'data/synthetic/3D_exciting_T2000.dill'
data = dill.load(open(dfile,'rb'))
d_spk, T, mu = data['spk'], data['T'], data['mu']
k_true = iclr2026_scenario.exciting()
n_node = len(k_true)

# Directory for Figures of Estimation Results ############
dir_result = './result_exciting_scenario/'
if not os.path.exists(dir_result):
    os.makedirs(dir_result)

# Settings of Cross-Validation for Hyper-parameter #######
# hyper-parameter candidates
set_g = [0.1, 0.5, 1.0] # "gamma": regularization
set_b = [0.5, 1.0, 1.5] # "beta": inverse scale
set_par = [[x,y] for x in set_g for y in set_b]

# support window
support = 5

# event data in [0,T*p_train] is used for fitting,
# and the rest for validation
p_train = 0.8

# Settings for Evaluation of Predictive Performance ######
t_ev = linspace(1.e-9,support,1000)
dt = diff(t_ev)
dt_ev = 0.5*r_[dt[0],(dt[1:]+dt[:-1]),dt[-1]]
ise, cpu = {'K2H':[]}, {'K2H':[]}
##########################################################

# Settings for Result Display ############################
x_lim, y_lim = [-0.1, support], [-0.1, 0.6]

for iii, spk in enumerate(d_spk):
    
    # Plot True Triggering Kernel ########################
    fig, ax = subplots(n_node,n_node,figsize=(12,6))
    for ii in range(n_node):
        for jj in range(n_node):
            ax[ii,jj].plot(t_ev,k_true[ii][jj](t_ev),'k--',lw=0.8)
            ax[ii,jj].set_xlim(x_lim[0],x_lim[1])
            ax[ii,jj].set_ylim(y_lim[0], y_lim[1])
        
    # Evaluation Points for Cross-Validation #############
    z = array([x[0] for x in spk])
    z = list(set(r_[z-1.e-9,z+1.e-9,linspace(0,T,10000)]))
    t_cr = array(sorted(z))
    
    dt = diff(t_cr)
    dt_cr = 0.5*r_[dt[0],(dt[1:]+dt[:-1]),dt[-1]]
    t_cr_tr = t_cr[where(t_cr<=T*p_train)]
    dt = diff(t_cr_tr)
    dt_cr_tr = 0.5*r_[dt[0],(dt[1:]+dt[:-1]),dt[-1]]

    # Split Data into Training and Test ##################
    spk_tr = []
    for x in spk:
        if x[0] <= T*p_train:
            spk_tr.append(x)

    # Format Data for Score Evaluation ###################
    def reform_spk(spk):
        n_node = max([x[1] for x in spk])
        y = [[] for x in range(n_node)]
        for x in spk:
            y[x[1]-1].append(x[0])
        return y
    spkk, spkk_tr = reform_spk(spk), reform_spk(spk_tr)
    
    # Initialize Proposed Method #########################
    model = 'K2H'
    k2h = k2_hawkes_rfm(kernel='gaussian', n_rand_feature=100)
    
    # Evaluate Predictive Performance (Least Squares Loss)
    # for Validation Data ################################
    set_score = []
    for (g,b) in set_par:
        
        _ = k2h.fit(spk_tr, T*p_train, gamma=g, b=b, support=support)
        rate = lambda x,s: [maximum(y,-99) for y in k2h.intensity(x,s)]
        
        score = np.sum(array(rate([t_cr]*n_node,spk))**2*dt_cr[None,:]) \
            - 2*sum([sum(x) for x in rate(spkk,spk)])
        score_tr = np.sum(array(rate([t_cr_tr]*n_node,spk_tr))**2*dt_cr_tr[None,:]) \
            - 2*sum([sum(x) for x in rate(spkk_tr,spk_tr)])

        set_score.append(score-score_tr)
            
    # Choose the Optimal Hyper-parameter #################
    indx = argmin(array(set_score))
    [opt_g, opt_b] = array(set_par)[indx]
    
    # Estimation with the Optimized Hyper-parameter ######
    t = k2h.fit(spk, T, gamma=opt_g, b=opt_b, support=support)

    # Plot Estimated Triggering Kernel ###################
    for ii in range(n_node):
        for jj in range(n_node):
            ax[ii,jj].plot(t_ev,k2h.predict(t_ev,edge=[ii,jj]),
                           'r',lw=0.8,alpha=0.8)

    # Evaluate CPU time and integrated squared error #####
    cpu[model].append(t)
    er = 0
    for ii in range(n_node):
        for jj in range(n_node):
            ker_ij = k2h.predict(t_ev,edge=[ii,jj])
            er += sum((k_true[ii][jj](t_ev)-ker_ij)**2*dt_ev)
    ise[model].append(er)
    
    savefig(dir_result+str(iii+1).zfill(2)+'.pdf')
    close('all')

    # Display Estimation Result ##########################
    print('')
    print('Trial = '+str(iii+1)+'/'+str(len(d_spk)))
    print('**ISE**')
    print(model+':',mean(ise[model]),std(ise[model]))
    print('**CPU**')
    print(model+':',mean(cpu[model]),std(cpu[model]))

