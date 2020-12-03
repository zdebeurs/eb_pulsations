get_ipython().magic('config InlineBackend.figure_format = "retina"')


#==========================================================================
#::: Standard workhorses
#==========================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#==========================================================================
#::: Exoplanet Package
#==========================================================================

# TEMPORARY WORKAROUND
import multiprocessing as mp

mp.set_start_method("fork")

import os
import logging
import warnings

# Don't use the schmantzy progress bar
os.environ["EXOPLANET_NO_AUTO_PBAR"] = "true"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="theano")

logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("exoplanet")
logger.setLevel(logging.DEBUG)

#==========================================================================
#::: Plotting
#==========================================================================
#For Zoom plots
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#Seaborn
sns.set(context='paper',  
	style='ticks',  
	palette='deep',  
	font='sans-serif', 
	font_scale=1.5, 
	color_codes=True,
	rc={
	"figure.dpi": 150,
	"savefig.dpi": 200,
	"axes.linewidth": 2,
	"lines.linewidth": 2, 
	"figure.figsize": (16,6), 
	"figure.dpi": 150, 
	"savefig.dpi": 200,
	"text.usetex": True, 
	"axes.labelsize": 30, ## fontsize of the x any y labels
	"xtick.labelsize": 25,
	"ytick.labelsize": 25})

sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})



'''

fontsize = 30
label_size = 20
tick_size = 25
linewidth = 2

plt.rcParams['font.titlesizeize'] = fontsize
plt.rcParams['axes.'] = label_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['figure.titlesize'] = fontsize
plt.rcParams['lines.linewidth'] = linewidth
plt.rcParams['axes.linewidth'] = linewidth

#Latex configuration
pgf_with_latex = {
    "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
    "text.usetex": True,                # use LaTeX to write all text
}
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update(pgf_with_latex)
'''
