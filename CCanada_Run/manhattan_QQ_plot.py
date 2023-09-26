import numpy as np
import pandas as pd
import magenpy as mgp
import viprs as vp
import subprocess
import io
from viprs.eval.metrics import r2 
from viprs.eval.metrics import pearson_r
from magenpy.plot import manhattan
from magenpy.plot import qq_plot
from magenpy.SumstatsTable import SumstatsTable
from Plot_utils import ELBO_plot
from Plot_utils import plot_obs_vs_pred
import matplotlib.pyplot as plt
# use grid search on validation data to find the suitable fixed parameters. 


# change the plink1.9 and plink2 path to local path
mgp.set_option("plink2_path", "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/plink/2.00a3.6/bin/plink2")
mgp.set_option("plink1.9_path","/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/plink/1.9b_6.21-x86_64/plink" )
mgp.print_options()

# real_phe for 2706 samples
region_gdl = mgp.GWADataLoader(
    bed_files = "CMAll_qced/merged10",  # CMAll_qced/merged10 haven't change and add CM
    phenotype_file = "data/phenotype_data/DREAM_pheno_Full922.csv",
    backend  = "plink",
    sumstats_files = "CCanada_Run/merge10.sumstats",
    sumstats_format = "magenpy"
)

# region_gdl.perform_gwas()   # only need plink2 (can use version: plink/2.00a3.6)
# region_gdl.to_summary_statistics_table().to_csv(
#     "CCanada_Run/merge10.sumstats", sep="\t", index=False
# )

region_gdl.harmonize_data()
gdl_sumstats = SumstatsTable(region_gdl.to_summary_statistics_table())
# draw manhattan plot
plt.figure()
manhattan(region_gdl, gdl_sumstats)
fig = plt.gcf()   # get the current figure 
fig.savefig("CCanada_Run/manhattan.png")



# draw the QQ plot
plt.figure()   # make the plt empty 
qq_plot(region_gdl, gdl_sumstats)
fig = plt.gcf()   # get the current figure 
fig.savefig("CCanada_Run/QQ_plot.png")
