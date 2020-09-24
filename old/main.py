#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:
Mariona Badenas-Agusti
Deparment of Earth, Atmospheric, and Planetary Sciences
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA  02139, 
USA
Email: mbadenas@mit.edu
"""
import logging
import warnings
import matplotlib.pyplot as plt
import exoplanet as xo
#import arviz as az
import eb_utils as utils
import eb_model as eb_model

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 16
plt.rcParams['font.sans-serif'] = ['Arial']
c_lc = 'lightsteelblue'

# Remove when Theano is updated
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Remove when arviz is updated
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("exoplanet")
logger.setLevel(logging.DEBUG)

    
if __name__ == '__main__':
    KIC = 4544587
    texp = 58.9 / 60. / 60. / 24.
    lit_period, lit_t0 = utils.get_true_eb_values(KIC) #values from EB catalog 
    
    # Download  Kepler data
    ntransits = 2 #nº of transits shown per star
    kepler_lc, tformat = utils.load_kepler_lc(KIC)
    
    #Prepare LC for MCMC fit
    cut_lc = utils.simplify_lc(kepler_lc, lit_period, tformat, ntransits)
    x, y, yerr, mu = utils.prepare_lc_for_mcmc(cut_lc)
    
    '''
    # Zoom Plot
    fig, ax = plt.subplots(1,1)
    cut_lc.scatter(marker='o',s=20,c=c_lc,ax=ax)
    ax.set(xlim=(811.5,812),ylim=(0.998,1.008))
    fig.savefig('figures/zoom_lc.png')
    
    # Plot Phase-folded LC
    fig, ax = plt.subplots(1,1)
    ax.scatter((x - lit_t0 + 0.5 * lit_period) % lit_period - 0.5 * lit_period, y, marker = ".", c = c_lc)
    ax.set(
        xlim = (-0.5 * lit_period, 0.5 * lit_period),
        xlabel = "Time since Primary Eclipse [days]", 
        ylabel = "SAP Flux [ppt]",
        title = 'Phase-folded LC (n = %i transits per star used)'%ntransits
    );
    ax.annotate(
        "Period = {0:.6f} days".format(lit_period),
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 5),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=15,
    );
    '''
    
    ################ ESTIMATE MAXIMUM A POSTERIORI PARAMETERS ################
    nloops = 1
    model, map_soln = eb_model.sigma_clip(nloops, x, y, lit_period, lit_t0, texp)
    
    #model predictions 
    with model:
        gp_pred = xo.eval_in_model(model.gp_lc.predict(), map_soln) + map_soln["mean_lc"]
        lc = xo.eval_in_model(model.model_lc(model.x), map_soln) - map_soln["mean_lc"]
   
    eb_model.plot_model_predictions(model, map_soln, lc, gp_pred) #plot model predictions

    '''
    ################ SAMPLING ################
    np.random.seed(42)
    with model:
        trace = xo.sample(
            tune=2000,
            draws=1500, #steps 
            start=map_soln,
            cores = 4, #from documentation: at most 4.
            chains = 4, #walkers || If None, then set to either cores or 2, whichever is larger.
            #initial_accept=0.8,
            target_accept=0.9,#step=xo.get_dense_nuts_step(target_accept=0.9)
        )
    
    idata = az.from_pymc3(trace, model=model) # dims=dims, coords=coords
    idata.to_netcdf('trace_results.nc')'''