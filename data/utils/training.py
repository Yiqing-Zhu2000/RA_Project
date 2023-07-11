import numpy as np
import pandas as pd
import magenpy as mgp
import viprs as vp
from viprs.eval.metrics import r2 

#simulate 2706 samples on 100 SNPs
def train_chr22_100SNPs():
    g_sim = mgp.GWASimulator("CMAll_qced/chr22/shuffle_100snps",
                            pi = [.99, .01],
                            h2=0.5)
    g_sim.simulate()
    g_sim.to_phenotype_table()
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
    g_sim.to_phenotype_table().to_csv("Toy_example_expr/phenotype/shuffle100_phe.txt",sep='\t', index=False, header=False)
    val_prs = v.predict(gdl_sim)
    return r2(val_prs, g_sim.sample_table.phenotype), v.history['ELBO']
    
    