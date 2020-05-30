#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:04:18 2020

@author:
Mariona Badenas-Agusti
Deparment of Earth, Atmospheric, and Planetary Sciences
Massachusetts Institute of Technology,
77 Massachusetts Avenue,
Cambridge, MA  02139, 
USA
Email: mbadenas@mit.edu

"""

import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az



if __name__ == '__main__':
    idata = az.from_netcdf("results/trace_results.nc")
    print(az.summary(idata))
    az.plot_pair(idata, var_names=["t0", "period", "R1", "M1", "ecs", "sigma"], kind="kde", fill_last=False);
    az.plot_posterior(idata, var_names="M1") #coords={"planet": ["55 Cnc e", "Kepler-37 c"]});