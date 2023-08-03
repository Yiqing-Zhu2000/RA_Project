import numpy as np
import pandas as pd
import magenpy as mgp
import viprs as vp
from viprs.eval.metrics import r2 
import matplotlib.pyplot as plt

def plot_obs_vs_pred(y_obs, x_pred):
    plt.scatter(x_pred, y_obs)
    plt.xlabel('predict phenotype')
    plt.ylabel('observed(real) phenotype')
    plt.title("Relation between obs.(real)& predict phe")
    plt.show()

def ELBO_plot(ELBO_list, save_path, itr):
    """
    params: a ELBO_list from the fitted viprs model v.history['ELBO']
    params: the abs./relative pathway end with .png 
    params: the itr " th" time that training again 
    """
    num = len(ELBO_list)
    plt.figure()
    plt.scatter(range(num), np.array(ELBO_list),s=15)
    plt.grid(which="major",alpha=0.3)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("Evidence lower bound as a function of EM outer itr " + str(itr) + "th")
    plt.savefig(save_path)
    plt.show()

#simulate 2706 samples on 100 SNPs
def train_chr22_100SNPs():
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/shuffle_100snps",
                            pi = [.95, .05],
                            h2=0.5)
    g_sim.simulate()
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.csv",sep='\t')
    # calculate gwas
    g_sim.perform_gwas()
    g_sim.to_summary_statistics_table().to_csv(
        "Toy_example_expr/shuffle_100snps.sumstats", sep="\t", index=False
    )
    # Load summary statistics（simulate phenotype from above) and match them with perviously
    gdl_sim = mgp.GWADataLoader(bed_files="CMAll_qced/chr22/shuffle_100snps",
                                sumstats_files="Toy_example_expr/shuffle_100snps.sumstats",
                                sumstats_format="magenpy")
    # calculate LD
    gdl_sim.compute_ld(estimator='sample',
                       output_dir='Toy_example_expr/shuffle100_chr22_out/')

    # gdl_sim.compute_ld(estimator='windowed',
    #                    output_dir='Toy_example_expr/ALLchr22_out/windowed/',
    #                    window_size=100)
    
    # training
    # viprs
    v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.001, 'sigma_epsilon': 0.999}) 
    v.fit()
    
    # predict on the same dataset directly 
    # g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.txt",sep='\t', index=False)
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype), v.history['ELBO']

    
def train_chr22_100SNPs_addCov():
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/shuffle_100snps",
                                pi = [.95, .05],
                                h2=0.5)
    g_sim.simulate()
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.csv",sep='\t')
    
    # input a phenotype and covariates, instead of .sumstats file 
    gdl_sim = mgp.GWADataLoader(bed_files="CMAll_qced/chr22/shuffle_100snps",
                                phenotype_file="Toy_example_expr/phenotype/shuffle100_phe.csv",
                                covariates_file="data/Dosage_for_PCA/chr22_covariates.csv",
                                )
    gdl_sim.perform_gwas()   # then the .sumstat file is calcuated and stored in model
    # compute LD
    gdl_sim.compute_ld(estimator='sample',
                    output_dir='Toy_example_expr/shuffle100_addCov_out/')
    
    # windows need to change output_dir if I need to use it. 
    # gdl_sim.compute_ld(estimator='windowed',
    #                    output_dir='Toy_example_expr/ALLchr22_out/windowed/',
    #                    window_size=100)
    
    # viprs
    v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.001, 'sigma_epsilon': 0.999})
    v.fit()
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype), v.history['ELBO']

def fixed_beta_100SNPs():
    beta100np = np.zeros(100)
    beta100np[20] = 0.5     # rs11090428
    beta100 = {22: beta100np}
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/shuffle_100snps",
                            pi = [.99, .01],
                            h2=0.5)
    g_sim.set_beta(beta100)
    g_sim.simulate(reset_beta=False)
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.csv",sep='\t')
    # calculate gwas
    g_sim.perform_gwas()
    g_sim.to_summary_statistics_table().to_csv(
        "Toy_example_expr/shuffle_100snps.sumstats", sep="\t", index=False
    )
    # Load summary statistics（simulate phenotype from above) and match them with perviously
    gdl_sim = mgp.GWADataLoader(bed_files="CMAll_qced/chr22/shuffle_100snps",
                                sumstats_files="Toy_example_expr/shuffle_100snps.sumstats",
                                sumstats_format="magenpy")
    # calculate LD
    gdl_sim.compute_ld(estimator='sample',
                        output_dir='Toy_example_expr/shuffle100_chr22_out/')
    # viprs
    v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.001, 'sigma_epsilon': 0.999}) 
    v.fit()

    # predict on the same dataset directly 
    # g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.txt",sep='\t', index=False)
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype),v.history['ELBO'],val_prs,g_sim.sample_table.phenotype
    
def fixed_beta_500Variants():
    beta500np = np.zeros(500)
    beta500np[0:2] = 0.5     # rs11090428
    beta500np[2:6] = -0.5
    beta500 = {22: beta500np}
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/shuffle_500snps",
                            pi = [.99, .01],
                            h2=0.5)
    g_sim.set_beta(beta500)
    g_sim.simulate(reset_beta=False)
    # g_sim.simulate()
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle500_phe.csv",sep='\t')
    # calculate gwas
    g_sim.perform_gwas()
    g_sim.to_summary_statistics_table().to_csv(
        "Toy_example_expr/shuffle_500snps.sumstats", sep="\t", index=False
    )
    # Load summary statistics（simulate phenotype from above) and match them with perviously
    gdl_sim = mgp.GWADataLoader(bed_files="CMAll_qced/chr22/shuffle_500snps",
                                sumstats_files="Toy_example_expr/shuffle_500snps.sumstats",
                                sumstats_format="magenpy")
    # calculate LD
    gdl_sim.compute_ld(estimator='sample',
                        output_dir='Toy_example_expr/shuffle500_chr22_out/')
    # viprs
    v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.001, 'sigma_epsilon': 0.999}) 
    # v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.998000, 'sigma_epsilon': 0.750000}) 
    v.fit()

    # predict on the same dataset directly 
    # g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.txt",sep='\t', index=False)
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype),v.history['ELBO'],val_prs,g_sim.sample_table.phenotype


#simulate 2706 samples
# this function haven't been tested.
def train_chr22ALL():
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/ALL_CM_chr22",
                            pi = [.99, .01],
                            h2=0.5)
    g_sim.simulate()
    g_sim.to_phenotype_table()
    # calculate gwas
    g_sim.perform_gwas()
    g_sim.to_summary_statistics_table().to_csv(
        "Toy_example_expr/ALL_CM22.sumstats", sep="\t", index=False
    )
    # Load summary statistics（simulate phenotype from above) and match them with perviously
    gdl_sim = mgp.GWADataLoader(bed_files="CMAll_qced/chr22/ALL_CM_chr22",
                                sumstats_files="Toy_example_expr/ALL_CM22.sumstats",
                                sumstats_format="magenpy")
    # calculate LD
    gdl_sim.compute_ld(estimator='sample',
                    output_dir='Toy_example_expr/ALLchr22_out/')

    # gdl_sim.compute_ld(estimator='windowed',
    #                    output_dir='Toy_example_expr/ALLchr22_out/windowed/',
    #                    window_size=100)
    
    # training
    # viprs
    v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.001, 'sigma_epsilon': 0.999}) 
    v.fit()
    
    # predict on the same dataset directly 
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/chr22ALL_phe.txt",sep='\t', index=False)
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype), v.history['ELBO']

def train_506_chr22():
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/Selected_1000snps",
                            pi = [.99, .01],
                            h2=0.5)
    g_sim.simulate()
    g_sim.to_phenotype_table()
    # calculate gwas
    g_sim.perform_gwas()
    g_sim.to_summary_statistics_table().to_csv(
        "Toy_example_expr/selected506.sumstats", sep="\t", index=False
    )
    # Load summary statistics（simulate phenotype from above) and match them with perviously
    gdl_sim = mgp.GWADataLoader(bed_files="CMAll_qced/chr22/Selected_1000snps",
                                sumstats_files="Toy_example_expr/selected506.sumstats",
                                sumstats_format="magenpy")
    # calculate LD
    gdl_sim.compute_ld(estimator='sample',
                    output_dir='Toy_example_expr/selected506_22_out/')

    # gdl_sim.compute_ld(estimator='windowed',
    #                    output_dir='Toy_example_expr/ALLchr22_out/windowed/',
    #                    window_size=100)
    
    # training
    # viprs
    v = vp.VIPRS(gdl_sim, fix_params={'pi': 0.001, 'sigma_epsilon': 0.999}) 
    v.fit()
    
    # predict on the same dataset directly 
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/selected506_22_phe.txt",sep='\t', index=False)
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype),v.history['ELBO'],val_prs,g_sim.sample_table.phenotype