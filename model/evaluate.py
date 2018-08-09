import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import sys

def get_ax():
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    return ax
    
def adjust_ax(ax):
    ax.set_xlim([0,1.05])
    ax.set_xticks([0,.2,.4,.6,.8,1.0])
    ax.set_ylim([0, 105])
    ax.set_ylabel('Percentage of Sim > x (%)')
    
def calc_AUC(cum_val):
    return np.sum(cum_val) / 100.0 / len(cum_val)

def sim_gt_val(sims, val_list, print_name = None, print_file = sys.stdout):
    percent_list = []
    for val in val_list:
        percent = 100.0*len(sims[sims>val]) / len(sims)
        if print_name: print("[R] {:.1f}% {}s > {:.2f}".format(percent, print_name ,val), file = print_file)
        percent_list.append(percent)
    return percent_list
    
def cum_plot(sim_list, sim_names, bin = 200, saveplot = None, print_file = sys.stdout):
    x_val = np.linspace(0,bin+1,bin+1)/bin
    df = pd.DataFrame(x_val,columns=['Sim = x'])
    ax = get_ax()
    auc_dict = {}
    for i in range(len(sim_list)):
        pcc = np.sort(sim_list[i])
        cum = [0] * (bin+1)
        for j in range(len(cum)):
            cum[j] = len(pcc[pcc>j/float(bin)]) / float(len(pcc)) * 100
        df[sim_names[i]] = cum
        auc_dict[sim_names[i]] = calc_AUC(cum)
        sim_gt_val(sim_list[i], [0.70, 0.75, 0.80, 0.85, 0.90], print_name = sim_names[i], print_file = print_file)
        print('', file = print_file)

    for i in range(len(sim_names)):
        df.plot(ax=ax,x='Sim = x', y=sim_names[i])
    adjust_ax(ax)
    if saveplot: ax.get_figure().savefig(saveplot)
    plt.clf()
    plt.cla()
    plt.close()
    return auc_dict