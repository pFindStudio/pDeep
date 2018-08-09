import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_ax():
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
    ax.set_ylabel('Percentage of PCC < x (%)')
    
def cum_plot(pcc_list, pcc_name, saveplot='../ipynb/figures/cum-pcc-EThcD.eps'):
    df = pd.DataFrame(np.linspace(0,201,201)/200,columns=['PCC = x'])
    ax = get_ax()
    for i in range(len(pcc_list)):
        pcc = np.sort(pcc_list[i])
        cum = [0] * 201
        for j in range(len(cum)):
            cum[j] = len(pcc[pcc<j/200.0]) / float(len(pcc)) * 100
        df[pcc_name[i]] = cum

    for i in range(len(pcc_name)):
        df.plot(ax=ax,x='PCC = x', y=pcc_name[i])
    adjust_ax(ax)
    ax.get_figure().savefig(saveplot)
