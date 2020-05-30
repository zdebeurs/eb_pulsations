#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:40:54 2020

@author:
Mariona Badenas-Agusti
Deparment of Earth, Atmospheric, and Planetary Sciences
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA  02139, 
USA
Email: mbadenas@mit.edu

"""
import pymc3 as pm
import theano.tensor as tt
import exoplanet as xo
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 16
plt.rcParams['font.sans-serif'] = ['Arial']
c_lc = 'lightsteelblue'

KIC = 4544587
    

def sigma_clip(nloops, x, y, lit_period, lit_t0, texp):
    mask = np.ones(len(x), dtype=bool)
    num = len(mask)
    for i in range(nloops):
        print("\n******************** Loop %i/%i ********************"%(i+1,nloops))
        model, map_soln = build_model(mask, x, y, lit_period, lit_t0, texp)
        with model:
            mdl = xo.eval_in_model(
            model.model_lc(x[mask]) + model.gp_lc.predict(), map_soln
        )

        resid = y[mask] - mdl
        sigma = np.sqrt(np.median((resid - np.median(resid)) ** 2))
        mask[mask] = np.abs(resid - np.median(resid)) < 7 * sigma
        print("Sigma clipped {0} light curve points".format(num - mask.sum()))
        if num == mask.sum():
            break
        num = mask.sum()

    return model, map_soln

def build_model(mask, x, y, lit_period, lit_t0, texp):
    with pm.Model() as model:
        # Systemic parameters
        mean_lc = pm.Normal("mean_lc", mu=0.0, sd=5.0) # mean flux of the time series

        u1 = xo.QuadLimbDark("u1")
        u2 = xo.QuadLimbDark("u2")
        
        # Parameters describing the primary
        M1 = pm.Lognormal("M1", mu=0.0, sigma=10.0)
        R1 = pm.Lognormal("R1", mu=0.0, sigma=10.0)

        # Secondary ratios
        k = pm.Lognormal("k", mu=0.0, sigma=10.0)  # radius ratio
        q = pm.Lognormal("q", mu=0.0, sigma=10.0)  # mass ratio
        s = pm.Lognormal("s", mu=np.log(0.5), sigma=10.0)  # surface brightness ratio
        
        # Parameters describing the orbit
        b = xo.ImpactParameter("b", ror=k, testval=1.5) 
        period = pm.Lognormal("period", mu=np.log(lit_period), sigma=1.0)
        t0 = pm.Normal("t0", mu=lit_t0, sigma=1.0) # The time of a reference transit
        
        # Parameters describing the eccentricity: ecs = [e * cos(w), e * sin(w)]
        ecs = xo.UnitDisk("ecs", testval=np.array([1e-5, 0.0]))
        ecc = pm.Deterministic("ecc", tt.sqrt(tt.sum(ecs ** 2)))
        omega = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0])) #arguments of periastron for the orbits in radians.

        # Build the orbit
        M2 = pm.Deterministic("M2", q * M1)
        R2 = pm.Deterministic("R2", k * R1)
        
        # Set up a Keplerian orbit for the EB
        orbit = xo.orbits.KeplerianOrbit(
            period=period,
            t0=t0,
            ecc=ecc,
            omega=omega, 
            b=b,
            r_star=R1,
            m_star=M1,
            m_planet=M2,
        )

        # Track some other orbital elements and model LC for plotting purposes
        pm.Deterministic("incl", orbit.incl)
        pm.Deterministic("a", orbit.a)

        # Noise model for the light curve
        sigma_lc = pm.InverseGamma("sigma_lc", testval=1.0, **xo.estimate_inverse_gamma_parameters(0.1, 10.0))        
        S_tot_lc = pm.InverseGamma("S_tot_lc", testval=2.5, **xo.estimate_inverse_gamma_parameters(1.0, 5.0))        
        #ell_lc = pm.InverseGamma("ell_lc", testval=2.0, **xo.estimate_inverse_gamma_parameters(1.0, 5.0))
        #kernel_lc = xo.gp.terms.SHOTerm(S_tot=S_tot_lc, w0=2 * np.pi / ell_lc, Q=1.0 / 3) 
        kernel_lc = xo.gp.terms.SHOTerm(S_tot=S_tot_lc, w0=lit_period, Q=1.0/np.sqrt(2)) 
        
        # Set up the LC secondary eclipse model computed using starry
        lc = xo.SecondaryEclipseLightCurve(u1, u2, s) 
        def model_lc(t):
            return (mean_lc + 1e3 * lc.get_light_curve(orbit=orbit, r=R2, t=t, texp=texp)[:, 0])

        gp_lc = xo.gp.GP(kernel_lc, x[mask], tt.zeros(mask.sum())**2 + sigma_lc**2, mean=model_lc)
        gp_lc.marginal("obs_lc", observed=y[mask])

        # Optimize the logp
        map_soln = model.test_point
        
        # First the LC parameters
        map_soln = xo.optimize(map_soln, [mean_lc, R1, k, s, b])
        map_soln = xo.optimize(map_soln, [mean_lc, R1, k, s, b, u1, u2])
        map_soln = xo.optimize(map_soln, [mean_lc, sigma_lc, S_tot_lc])#, ell_lc])
        map_soln = xo.optimize(map_soln, [t0, period])

        # # Optimize to find the maximum a posteriori parameters
        map_soln = xo.optimize(map_soln)

        model.gp_lc = gp_lc
        model.model_lc = model_lc
        model.x = x[mask]
        model.y = y[mask]

        return model, map_soln

def plot_model_predictions(model,map_soln, lc, gp_pred):
    #At the best fit parameters, let's make some plots of the model predictions 
    #compared to the observations to make sure that things look reasonable.
    
    #GP for the phase curve. 
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 7))
    fig.subplots_adjust(hspace=0.1)
    ax1.plot(model.x, model.y, c=c_lc, marker = '.', alpha=0.2, label = 'data')
    ax1.plot(model.x, gp_pred, color="C1", lw=1, label = 'GP model')
    ax1.legend(loc='upper right')
    ax1.set(ylabel = "Raw flux [ppt]", title = 'KIC%s - Map Model for GP'%KIC)
    
    ax2.plot(model.x, model.y - gp_pred,  c=c_lc, marker='.', alpha=0.2, label = 'data - GP model')
    ax2.plot(model.x, lc, color="crimson", lw=1)
    ax2.legend(loc='upper right')
    ax2.set(
        xlim = (model.x.min(), model.x.max()),
        xlabel = "Time [BKJD days]", 
        ylabel = "De-trended flux [ppt]"
    );
    fig.savefig('figures/map_model_1.png');
    
    #Phase folded light curve.
    fig, ax = plt.subplots(1, figsize=(12, 3.5))
    x_fold = (model.x - map_soln["t0"]) % map_soln["period"] / map_soln["period"]
    inds = np.argsort(x_fold)
    ax.scatter(x_fold[inds], model.y[inds] - gp_pred[inds], c=c_lc, marker='.')
    ax.scatter(x_fold[inds] - 1, model.y[inds] - gp_pred[inds], c=c_lc, marker='.')
    ax.scatter(x_fold[inds], lc[inds], c='salmon',marker = '.',lw=0.5, alpha=0.3)
    ax.scatter(x_fold[inds] - 1, lc[inds], c='salmon',marker = '.',lw=0.5, alpha=0.3)
    ax.set(
        xlim=(-1, 1), 
        ylabel = "De-trended Flux [ppt]", 
        xlabel="Phase [d]",
        title = "KIC%s - Map Model for Phase-Folded LC" %KIC
    );
    plt.show()
    fig.savefig('figures/map_model_2.png')
    plt.close("all")
