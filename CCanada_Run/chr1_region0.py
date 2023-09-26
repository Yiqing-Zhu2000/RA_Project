import numpy as np
import pandas as pd
import magenpy as mgp
import viprs as vp
import subprocess
import io
from viprs.eval.metrics import r2 
from viprs.eval.metrics import pearson_r
from Plot_utils import ELBO_plot
from Plot_utils import plot_obs_vs_pred
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr

# chr1 region0 
np.random.seed(1235)
# paths need to be updated for different chr. 
# use to compute LD for training set.
i = 0 # region
chr_bed_filepath = "CMAll_qced/binary_chr1/region_snps/region"+str(i)+"/region"+str(i)
sumstats_path = "CMAll_qced/binary_chr1/region_snps/region"+str(i)+"/region"+str(i)+".sumstats"
output_LD_dirpath = "CMAll_qced/binary_chr1/region_snps/region"+str(i)+"/"   # ld/chr1 would been built there 
ELBO_plot_path = "CMAll_qced/binary_chr1/region_snps/region"+str(i)+"/elbo.png"
M_fixed_paras = {'pi':0.06, 'sigma_epsilon':  0.95}  # 'pi':0.03,

# real_phe for 2706 samples
region_gdl = mgp.GWADataLoader(
    bed_files = chr_bed_filepath,
    phenotype_file = "data/phenotype_data/DREAM_pheno_Full.csv",
    # backend  = "plink",
    sumstats_files= "CMAll_qced/binary_chr1/region_snps/region0/region0.sumstats",  # gwas put here
    sumstats_format= "magenpy",
)

Train_region_gdl, Val_region_gdl = region_gdl.split_by_samples(proportions=[.8, .2])

# Train_region_gdl.perform_gwas()   # only need plink2 (can use version: plink/2.00a3.6)
# Train_region_gdl.to_summary_statistics_table().to_csv(
#     sumstats_path, sep="\t", index=False
# )


# Train_region_gdl.compute_ld(estimator='sample',
#                 output_dir=output_LD_dirpath)
Train_region_gdl.read_ld("CMAll_qced/binary_chr1/region_snps/region0/ld/chr_1")

v_reg = vp.VIPRS(Train_region_gdl, fix_params = M_fixed_paras ) # 
# v = vp.VIPRS(realA22Train_gdl) 
# theta_0 = {'pi': 0.999945, 'sigma_epsilon': 0.50}
# v.initialize(theta_0=theta_0)
v_reg.fit()
print("after fit now.")

ELBO_plot(v_reg.history['ELBO'], ELBO_plot_path,itr=0)

print("It works!")


# predict and compute the R value. 
pred = v_reg.predict(Val_region_gdl)
print("val R:", pearson_r(pred, Val_region_gdl.sample_table.phenotype))

pred_ytrain = v_reg.predict(Train_region_gdl)
print("train R:", pearson_r(pred_ytrain, Train_region_gdl.sample_table.phenotype))

print("the Val_gdl real phe vs. predict phe")
plot_obs_vs_pred(Val_region_gdl.sample_table.phenotype, pred)