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

# change the plink1.9 and plink2 path to local path
mgp.set_option("plink2_path", "/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/plink/2.00a3.6/bin/plink2")
mgp.set_option("plink1.9_path","/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/plink/1.9b_6.21-x86_64/plink" )
mgp.print_options()

np.random.seed(1235)
# paths need to be updated for different chr. 
# use to compute LD for training set.

num_chr = 22
chr_bed_filepath = "CMAll_qced/binary_chr"+str(num_chr)+"/ALL_chr"+str(num_chr)
sumstats_path = "CMAll_qced/binary_chr"+str(num_chr)+"/plink_real_chr"+str(num_chr)+".sumstats"
output_LD_dirpath = "CMAll_qced/binary_chr"+str(num_chr)+"/"   # ld/chr1 would been built there 
ELBO_plot_path = "CMAll_qced/binary_chr"+str(num_chr)+"/ELBO.png"
# store_predVal = "CMAll_qced/binary_chr"+str(num_chr)+"/val_predict.npy"
# store_predTrain = "CMAll_qced/binary_chr"+str(num_chr)+"/train_predict.npy"
M_fixed_paras = {'pi':0.05, 'sigma_epsilon':  0.70}  # 'pi':0.03,

# real_phe for 2706 samples
region_gdl = mgp.GWADataLoader(
    bed_files = chr_bed_filepath,
    phenotype_file = "data/phenotype_data/DREAM_pheno_Full922.csv",
    backend  = "plink",
    ld_store_files = "CMAll_qced/binary_chr22/ld/chr_22",
    # sumstats_files= "CMAll_qced/binary_chr22/plink_real_chr22.sumstats",  # gwas put here
    # sumstats_format= "magenpy",
    # covariates_file = "data/covariates/NoDos_covHasHeader.csv",
    
)

Train_region_gdl, Val_region_gdl = region_gdl.split_by_samples(proportions=[.8, .2])

Train_region_gdl.perform_gwas()   # only need plink2 (can use version: plink/2.00a3.6)
Train_region_gdl.to_summary_statistics_table().to_csv(
    sumstats_path, sep="\t", index=False
)

# harmonization is really important to 
Train_region_gdl.harmonize_data()

v_reg = vp.VIPRS(Train_region_gdl, fix_params = M_fixed_paras ) # 
v_reg.fit()

ELBO_plot(v_reg.history['ELBO'], ELBO_plot_path,itr=0)


print("use the fixed params:", M_fixed_paras)
# predict and compute the R value. 
pred = v_reg.predict(Val_region_gdl)
print("val R:", pearson_r(pred, Val_region_gdl.sample_table.phenotype))
print("val R^2", r2(pred, Val_region_gdl.sample_table.phenotype))

pred_ytrain = v_reg.predict(Train_region_gdl)
print("train R:", pearson_r(pred_ytrain, Train_region_gdl.sample_table.phenotype))
print("train R^2:", r2(pred_ytrain, Train_region_gdl.sample_table.phenotype))


# pred_ytest = v_reg.predict(test_gdl)
# print("test R:", pearson_r(pred_ytest, test_gdl.sample_table.phenotype))
# print("test R^2:", r2(pred_ytest, test_gdl.sample_table.phenotype))
