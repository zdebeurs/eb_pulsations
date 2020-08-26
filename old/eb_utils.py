#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:26:19 2020

@author:
Mariona Badenas-Agusti
Deparment of Earth, Atmospheric, and Planetary Sciences
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA  02139, 
USA
Email: mbadenas@mit.edu

"""
import numpy as np
import lightkurve as lk
import statistics
import pandas as pd

def get_texp(name,cadence_type):
    tpf = lk.search_targetpixelfile(
        "KIC%s"%name, 
        cadence = cadence_type,
        mission = 'kepler').download();
    
    hdr = tpf.hdu[1].header
    texp = hdr["FRAMETIM"] * hdr["NUM_FRM"]
    texp /= 60.0 * 60.0 * 24.0
    return texp

def load_kepler_lc(name):
    data = lk.search_lightcurvefile(
        'KIC%s'%name, 
        cadence = 'short',
        mission = 'kepler').download_all()
    
    lc_data = data.SAP_FLUX.stitch().remove_nans().remove_outliers(sigma = 3, sigma_lower = float('inf'))
    tformat = lc_data.time_format
    return lc_data, tformat

def simplify_lc(lc, P, tformat, ntransits):
    middle_value = statistics.median(lc.time) 
    ndays = np.round(ntransits*P)
    mask_binning = (lc.time>middle_value) & (lc.time<middle_value+ndays)

    cut_lc = lk.LightCurve(
        time=lc.time[mask_binning == True], 
        flux=lc.flux[mask_binning == True], 
        flux_err = lc.flux_err[mask_binning == True], 
        time_format = tformat)
    return cut_lc
    
def prepare_lc_for_mcmc(lc):
    lc_x = np.ascontiguousarray(lc.time, dtype=np.float64)
    lc_y = np.ascontiguousarray(lc.flux, dtype=np.float64)
    lc_yerr = np.ascontiguousarray(lc.flux_err, dtype=np.float64)
    lc_mu = np.median(lc_y)
    lc_y = (lc_y / lc_mu - 1) * 1e3 #Units: Relative Flux [ppt]
    lc_yerr = lc_yerr * 1e3 / lc_mu
    return lc_x, lc_y, lc_yerr, lc_mu

def get_true_eb_values(name):
    kepler_tref = 2454833
    villanova_tref = 2400000 
    
    catalog = pd.read_csv('catalog_of_EBs_with_SC.csv', sep = ",", index_col = '#KIC')
    P =  catalog.loc[name, 'period']
    t0 =  catalog.loc[name, 'bjd0']+villanova_tref-kepler_tref # Time of eclipse. Convention s.t. the primary (deeper) eclipse occurs at phase 0 (BJD-2400000)'
    return P, t0
    
