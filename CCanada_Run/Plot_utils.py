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