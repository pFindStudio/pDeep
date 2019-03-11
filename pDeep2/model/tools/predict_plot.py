#coding=utf-8
'''
Created on 2013.12.13

@author: dell
'''

import numpy as np
from scipy.stats.stats import pearsonr

import matplotlib.pyplot as plt
#from matplotlib.figure import Figure
from matplotlib.pyplot import figure as Figure
import os
import struct

from ..featurize import Ion2Vector
from .ion_calc import *

mutation_table = {
    "Ala" : "A",
    "Cys" : "C",
    "Asp" : "D",
    "Glu" : "E",
    "Phe" : "F",
    "Gly" : "G",
    "His" : "H",
    "Ile" : "I",
    "Lys" : "K",
    "Leu" : "L",
    "Met" : "M",
    "Asn" : "N",
    "Pro" : "P",
    "Gln" : "Q",
    "Arg" : "R",
    "Ser" : "S",
    "Thr" : "T",
    "Val" : "V",
    "Trp" : "W",
    "Xle" : "L",
    "Tyr" : "Y",
}

class MS2s(object):
    def __init__(self):
        self.spectra = {}

def read_section_until(fin, sec_begin, sec_end):
    sections = []
    line = fin.readline()
    while not line.startswith(sec_begin):
        if line == "": return []
        line = fin.readline()
    while not line.startswith(sec_end):
        if line == "": return []
        if len(line) >= 2: 
            sections.append(line.strip())
        line = fin.readline()
    return sections

def get_value_from_section(sections, key):
    for sec in sections:
        if sec.startswith(key):
            return sec[sec.find("=")+1:]
    return ""

def get_ax(fig):
    ax = fig.add_subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    return ax
    
# class MS2(object):
    # def __init__(self):
        # self.name = ""
        # self.charge = 0
        # self.pepmass = 0
        # self.tPeakIdxinMGF = 0
        
    # def ReadMzIntsByIdx(self, mgf_file):
        # mgf_file = open(mgf_file)
        # mgf_file.seek(self.tPeakIdxinMGF)
        # mz_ints = []
        # line = mgf_file.readline()
        # while not line.startswith("END"):
            # if len(line) < 2 or not line[0].isdigit():
                # line = mgf_file.readline()
                # continue
            # mz_int = line.split()
            # mz = float(mz_int[0])
            # intens = float(mz_int[1])
            # mz_ints.append((mz,intens))
            # line = mgf_file.readline()
        # mgf_file.close()
        # mz_ints.sort(key = lambda x: x[0])
        # return mz_ints

# class MGFReader(object):
    # '''
    # classdocs
    # '''

    # def __init__(self, mgf):
        # '''
        # Constructor
        # '''
        
        # self.ms2s = MS2s()
        # self.reader = open(mgf,"r")
        
    # def ReadOneSpectrum(self):
        # ms2 = MS2()
        # sections = read_section_until(self.reader, "BEGIN IONS", "END IONS")
        # if len(sections) == 0: return False
        # ms2.name = get_value_from_section(sections, "TITLE")
        # ms2.pepmass = float(get_value_from_section(sections, "PEPMASS"))
        # charge = get_value_from_section(sections, "CHARGE")
        # if charge != "":
            # ms2.charge = int(charge[:charge.find('+')])
        # else:
            # ms2.charge = 1
        # self.ms2s.spectra[ms2.name] = ms2
        # return True
    
    # def ReadAll(self):
        # self.ms2s = MS2s()
        # ret = self.ReadOneSpectrum()
        # while ret != False:
            # ret = self.ReadOneSpectrum()
        # self.reader.close()
        

class pdeep_plot(object):
    '''
    classdocs
    '''
    #
    def __init__(self, conf, pdeep_model, plot_ion_types, tol = 20, tol_type = "ppm"):
        '''
        Constructor
        '''
        
        # self.mass_H = 1.0078250321
        # self.mass_O = 15.9949146221
        self.mass_proton = aamass.mass_proton
        # self.mass_N = 14.0030740052
        # self.mass_C = 12.00
        # self.mass_isotope = 1.003
        
        # self.mass_H2O = self.mass_H * 2 + self.mass_O
        # self.mass_CO = self.mass_C + self.mass_O
        # self.mass_CO2 = self.mass_C + self.mass_O * 2
        # self.mass_NH = self.mass_N + self.mass_H
        # self.mass_NH3 = self.mass_N + self.mass_H * 3
        # self.mass_HO = self.mass_H + self.mass_O
        
        # self.aa_mass_dict = {}
        # self.aa_mass_dict['A'] = 71.037114
        # self.aa_mass_dict['C'] = 103.009185# + 57.021464
        # self.aa_mass_dict['D'] = 115.026943
        # self.aa_mass_dict['E'] = 129.042593
        # self.aa_mass_dict['F'] = 147.068414
        # self.aa_mass_dict['G'] = 57.021464
        # self.aa_mass_dict['H'] = 137.058912
        # self.aa_mass_dict['I'] = 113.084064
        # self.aa_mass_dict['J'] = 114.042927
        # self.aa_mass_dict['K'] = 128.094963
        # self.aa_mass_dict['L'] = 113.084064
        # self.aa_mass_dict['M'] = 131.040485
        # self.aa_mass_dict['N'] = 114.042927
        # self.aa_mass_dict['P'] = 97.052764
        # self.aa_mass_dict['Q'] = 128.058578
        # self.aa_mass_dict['R'] = 156.101111
        # self.aa_mass_dict['S'] = 87.032028
        # self.aa_mass_dict['T'] = 101.047679
        # self.aa_mass_dict['U'] = 150.95363
        # self.aa_mass_dict['V'] = 99.068414
        # self.aa_mass_dict['W'] = 186.079313
        # self.aa_mass_dict['Y'] = 163.06332
        # self.mod_mass_dict = {}
        # self.mod_mass_dict["Carbamidomethyl[C]"] = (57.021464, 0)
        # self.mod_mass_dict['Oxidation[M]'] = (15.994915, 63.998285)
        # self.__read_mod__()
        
        self.tol = tol
        self.tol_type = tol_type
        self.mgfreader = None
        self.mz_bin_size = 0.02
        if self.tol_type.upper() == "DA":
            self.mz_bin_size = self.tol
        self.max_plot_mz = 2100
        self.min_plot_inten = 1e-10
        
        self.ion_color = {'b{}':'#1E1EE5','y{}':'#E51E1E','c{}':'darkgreen','z{}':'#9370db','b{}-ModLoss':'#1E1EE57F','y{}-ModLoss':'#E51E1E7F','b{}-H2O':'#1E1EE5','y{}-H2O':'#E51E1E','b{}-NH3':'#1E1EE5','y{}-NH3':'#E51E1E'}
        self.config = conf
        self.ion_types = plot_ion_types
        
        self.model = pdeep_model
        self.ion2vec = Ion2Vector(self.config)
        
        self.show_plot = True
    
    def Plot_Predict(self, plotax, ions, predictions):
        valign = 'bottom'
        vmargin = 0.05
        
        max_charge = self.config.max_ion_charge if self.spec_charge >= self.config.max_ion_charge else self.spec_charge
        matched_inten = []
        for charge in range(1, max_charge+1):
            for ion_type in self.ion_types:
                target_ions = ions[ion_type]
                for i in range(len(target_ions)):
                    x = target_ions[i] / charge + self.mass_proton
                    y = predictions[i, self.config.GetIonIndexByIonType(ion_type, charge)]
                    if x > self.max_plot_mz or y < self.min_plot_inten or x < 10: #x < 2Da for modloss outside the modsite
                        matched_inten.append(0)
                    else:
                        if self.config.ion_terms[ion_type] == 'c': ion_idx = len(target_ions)-i
                        else: ion_idx = i+1
                        if self.show_plot:
                            plotax.plot( [x,x], [0, y], color=self.ion_color[ion_type], lw=2)
                            plotax.text( x, y + vmargin, ion_type.format(str(ion_idx)+'+'*charge), rotation = 90, color=self.ion_color[ion_type], horizontalalignment="center",verticalalignment=valign)
                        matched_inten.append(y)
        return matched_inten
    
    def output_predict_with_iontype(self, peptide, modinfo, charge):
        x, mod_x = self.ion2vec.FeaturizeOnePeptide(peptide, modinfo, charge)
        if x is None: return
        if self.mod_mode:
            prediction = self.model.predict([np.array([x]),np.array([mod_x])])[0,:,:]
        else:
            prediction = self.model.predict(np.array([x]))[0,:,:]
        
        pred_charge = charge-1 if charge <= self.config.max_ion_charge else self.config.max_ion_charge
        output = {}
        for i in range(len(self.config.ion_types)):
            it = self.config.ion_types[i]
            for ch in range(1, pred_charge+1):
                output['{}+{}'.format(ch, it)] = prediction[:len(peptide)-1, i*self.config.max_ion_charge + ch-1]
        return output
    
    def plot(self, peptide, modinfo, charge, nce, instrument):
        ions = {}
        bions, pepmass = calc_b_ions(peptide, modinfo)
        print("parent m/z (%d+) = %.6f" %(charge, pepmass/charge + self.mass_proton))
        if 'b{}' in self.config.ion_types: ions['b{}'] = bions
        if 'y{}' in self.config.ion_types: ions['y{}'] = calc_y_from_b(bions, pepmass)
        if 'c{}' in self.config.ion_types: ions['c{}'] = calc_c_from_b(bions)
        if 'z{}' in self.config.ion_types: ions['z{}'] = calc_z_from_b(bions, pepmass)
        if 'b{}-ModLoss' in self.config.ion_types: ions['b{}-ModLoss'] = calc_ion_modloss(bions, peptide, modinfo, N_term = True)
        if 'y{}-ModLoss' in self.config.ion_types: ions['y{}-ModLoss'] = calc_ion_modloss(ions['y{}'], peptide, modinfo, N_term = False)
        
        max_plot_mz = min(self.max_plot_mz, max([max(tmp_ions) for tmp_ions in ions.values()])+200)
        
        if self.show_plot: 
            fig = Figure(figsize=(12,8))
            ax = get_ax(fig)
        else:
            fig = None
            ax = None
        self.spec_charge = charge
        self.pepmass = pepmass
        
        buckets = self.ion2vec.FeaturizeOnePeptide_buckets(peptide, modinfo, self.spec_charge, nce, instrument)
        if buckets is None: return
        predictions = self.model.Predict(buckets)[len(peptide)][0][0,:,:]
        matched_inten2 = self.Plot_Predict(ax, ions, predictions)
        
        if self.show_plot:
            ax.text(200, 1.3, '{} ({}+), {}'.format(peptide,self.spec_charge,modinfo), fontsize = 14)
            
            ax.set_xlim(xmin = 0, xmax = max_plot_mz)
            ax.set_ylim(ymin = 0, ymax=1.4)
            ax.hlines([0], [0], [max_plot_mz])
            ax.set_xlabel('m/z')
            ax.set_ylabel('Relative Abundance')
            ylabels = ax.get_yticks().tolist()
            # ylabels = ["{:.0f}%".format(abs(float(label)*100)) if float(label) >= 0 else '' for label in ylabels]
            ylabels = ["{:.0f}%".format(abs(float(label)*100)) for label in ylabels]
            #ylabels = ['' for label in ylabels]
            ax.set_yticklabels(ylabels)
            
        return fig
        
    def batch_save(self, pf2, pep_list, save_dir):
        self.Readpf2(pf2)
        for pep in pep_list:
            
            def mutation_and_short_mod(peptide, mod):
                # ACDEFMGK + 1,Ala->Val[A];6,Oxidation; ==> return VCDEFMGK, 6,Oxidation, 6M
                
                def short_a_moditem(item):
                    return item[:item.find(",")] + item[item.rfind("[")+1:item.rfind("]")]
                
                if mod == "": return peptide, mod, mod
                short_name = ""
                new_mod = []
                items = mod.strip(";").split(";")
                for item in items:
                    short_name += short_a_moditem(item)
                    idx = item.find("->")
                    if idx != -1:
                        mut_to = item[idx+2:item.find("[")]
                        if mut_to in mutation_table:
                            site = int(item[:item.find(",")]) - 1
                            peptide = peptide[:site] + mutation_table[mut_to] + peptide[site+1:]
                        else:
                            new_mod.append(item)
                    else:
                        new_mod.append(item)
                
                return peptide, ";".join(new_mod), short_name
             
            peptide, mod, short_mod = mutation_and_short_mod(pep[0], pep[1])
            
            fig, pcc = self.plot(peptide, mod, pep[2], pep[3], pep[4])
            plt.tight_layout()
            mng = plt.get_current_fig_manager()
            #mng.window.showMaximized()
            #mng.resize(*mng.window.maxsize())
               
            
            plt.savefig(os.path.join(save_dir, "%s-%s-%s-R=%.3f.png"%(pep[2], peptide, short_mod, pcc)),format="png", dpi=220)
            plt.close()
            
    def batch_show(self, pep_list, save_dir = None):
        for pep in pep_list:
            self.plot(*pep)
            self.show()
    
    def show(self):
        if self.show_plot: plt.show()

#if __name__ == "__main__":
    # jupyter usage
    # from model.tools.back2back_plot import bbplot
    # import model.fragmentation_config as fconfig
    # import model.lstm_unmod as lstm

    # lstm_unmod = lstm.IonLSTM(fconfig.HCD_Config())
    # lstm_unmod.LoadModel(r'.\h5\len20\2-layer-BiLSTM-QE-M-M.h5')

    # bb = bbplot(conf = fconfig.HCD_Config(), model = lstm_unmod)
    # bb.Readpf2(r'h:\zhouxiexuan\data2RawFiles\HCD\RAW\01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1_HCDFT.pf2')
