import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import sys

thres_list = [0.7, 0.75, 0.8, 0.85, 0.9]

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
    
def auc_name(sim_name):
    return sim_name + "-AUC"
def med_name(sim_name):
    return sim_name + "-Med"
def thres_name(sim_name, thres):
    return "%s%d"%(sim_name, int(100*thres))

def sim_gt_val(sims, val_list, print_name = None, print_file = sys.stdout):
    percent_dict = {}
    for val in val_list:
        if len(sims) > 0: percent = len(sims[sims>val]) / len(sims)
        else: percent = 0
        percent_dict[thres_name(print_name, val)] = percent
        if print_name: print("[R] {:.1f}% {}s > {:.2f}".format(100.0*percent, print_name ,val), file = print_file)
    return percent_dict
    
def median_val(sims, print_name = None, print_file = sys.stdout):
    print("[R] {:.3f} median {}".format(np.median(sims), print_name), file = print_file)
    return (med_name(print_name), np.median(sims))
    
def cum_plot(sim_list, sim_names, thres_list, bin = 200, saveplot = None, print_file = sys.stdout):
    x_val = np.linspace(0,bin+1,bin+1)/bin
    df = pd.DataFrame(x_val,columns=['Sim = x'])
    ax = get_ax()
    result_dict = {}
    for i in range(len(sim_list)):
        pcc = np.sort(sim_list[i])
        cum = [0] * (bin+1)
        for j in range(len(cum)):
            if len(pcc) > 0: cum[j] = len(pcc[pcc>j/float(bin)]) / float(len(pcc)) * 100
            else: cum[j] = 0
        df[sim_names[i]] = cum
        result_dict[auc_name(sim_names[i])] = calc_AUC(cum)
        sim_dict = sim_gt_val(sim_list[i], thres_list, print_name = sim_names[i], print_file = print_file)
        name, med = median_val(sim_list[i], print_name = sim_names[i], print_file = print_file)
        print('', file = print_file)
        
        result_dict.update(sim_dict)
        result_dict[name] = med

    for i in range(len(sim_names)):
        df.plot(ax=ax,x='Sim = x', y=sim_names[i])
    adjust_ax(ax)
    if saveplot: ax.get_figure().savefig(saveplot)
    plt.clf()
    plt.cla()
    plt.close()
    return result_dict