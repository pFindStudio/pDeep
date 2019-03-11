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
        

class bbplot(object):
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
        self.min_plot_inten = 1e-4
        
        self.ion_color = {'b{}':'#1E1EE5','y{}':'#E51E1E','c{}':'darkgreen','z{}':'#9370db','b{}-ModLoss':'#1E1EE57F','y{}-ModLoss':'#E51E1E7F','b{}-H2O':'#1E1EE5','y{}-H2O':'#E51E1E','b{}-NH3':'#1E1EE5','y{}-NH3':'#E51E1E'}
        self.config = conf
        self.ion_types = plot_ion_types
        
        self.model = pdeep_model
        self.ion2vec = Ion2Vector(self.config)
        
        self.show_plot = True
    
    def CalcFragmentTol(self, mz):
        if self.tol_type.upper() == "DA":
            return self.tol
        else:
            return self.tol * mz / 1000000.0
        
    def Readpf2(self, pf2_file):
        self.pf2_file = pf2_file
        self.raw_name = os.path.basename(pf2_file)
        if self.raw_name.rfind("_") != -1:
            self.raw_name = self.raw_name[:self.raw_name.rfind("_")]
        else:
            self.raw_name = self.raw_name[:self.raw_name.rfind(".")]
        self.__open_pf2__(pf2_file)
        self.__read_pf2idx__(pf2_file + "idx")
    
    def __read_pf2idx__(self, pf2idx_file):
        self.pf2idx = {}
        f = open(pf2idx_file,'rb')
        while True:
            chunk = f.read(8)
            if not chunk: break
            scan_no, index = struct.unpack('2i',chunk)
            self.pf2idx['{}.{}'.format(self.raw_name, scan_no)] = index
        f.close()
    
    def __open_pf2__(self, pf2_file):
        self.pf2 = open(pf2_file,'rb')
    
    def __read_one_spec__(self, raw_scan):
        self.pf2.seek(self.pf2idx[raw_scan])
        scan_no, nPeak = struct.unpack("2i",self.pf2.read(8))
        mz_int = struct.unpack(str(nPeak*2)+"d", self.pf2.read(nPeak*2*8))
        peaks = []
        for i in range(nPeak):
            mz = mz_int[i*2]
            inten = mz_int[i*2+1]
            peaks.append( (mz, inten) )
        return peaks
        
    def peak_hashing(self, peaks):
        max_mz, __ = max(peaks, key = lambda x: x[0])
        self.hash_table = [[] for i in range(int(max_mz / self.mz_bin_size)+10)]
        for i in range(len(peaks)):
            self.hash_table[int(peaks[i][0] / self.mz_bin_size)].append(i)
            
    def match_use_hashing(self, ions, charge, peaks):
        matched_peak_idx = [-1] * len(ions)
        matched_peak_inten = [0] * len(ions)
        for i in range(len(ions)):
            mass = ions[i] / charge + self.mass_proton
            if (mass / self.mz_bin_size) > len(self.hash_table) - 5: continue 
            min_tol = self.CalcFragmentTol(mass)
            hashed_peak_idlist = self.hash_table[int(mass / self.mz_bin_size)]
            for idx in hashed_peak_idlist:
                if abs(peaks[idx][0] - mass) <= min_tol:
                    matched_peak_idx[i] = idx
                    matched_peak_inten[i] = peaks[idx][1]
                    min_tol = abs(peaks[idx][0] - mass)
            
            # check the bound out of the bin
            hashed_peak_idlist = self.hash_table[int(mass / self.mz_bin_size)-1]
            for idx in hashed_peak_idlist:
                if abs(peaks[idx][0] - mass) <= min_tol:
                    matched_peak_idx[i] = idx
                    matched_peak_inten[i] = peaks[idx][1]
                    min_tol = abs(peaks[idx][0] - mass)
            
            hashed_peak_idlist = self.hash_table[int(mass / self.mz_bin_size)+1]
            for idx in hashed_peak_idlist:
                if abs(peaks[idx][0] - mass) <= min_tol:
                    matched_peak_idx[i] = idx
                    matched_peak_inten[i] = peaks[idx][1]
                    min_tol = abs(peaks[idx][0] - mass)
        
        return matched_peak_idx, matched_peak_inten
    
    def Plot_Predict(self, plotax, ions, predictions):
        valign = 'top'
        vmargin = -0.05
        
        predictions = -predictions*self.max_real_inten
        max_charge = self.config.max_ion_charge if self.spec_charge >= self.config.max_ion_charge else self.spec_charge
        matched_inten = []
        for charge in range(1, max_charge+1):
            for ion_type in self.ion_types:
                target_ions = ions[ion_type]
                for i in range(len(target_ions)):
                    x = target_ions[i] / charge + self.mass_proton
                    y = predictions[i, self.config.GetIonIndexByIonType(ion_type, charge)]
                    if x > self.max_plot_mz or y > -self.min_plot_inten or x < 10: #x < 2Da for modloss outside the modsite
                        matched_inten.append(0)
                    else:
                        if self.config.ion_terms[ion_type] == 'c': ion_idx = len(target_ions)-i
                        else: ion_idx = i+1
                        if self.show_plot:
                            plotax.plot( [x,x], [0, y], color=self.ion_color[ion_type], lw=2)
                            plotax.text( x, y + vmargin, ion_type.format(str(ion_idx)+'+'*charge), rotation = 90, color=self.ion_color[ion_type], horizontalalignment="center",verticalalignment=valign)
                        matched_inten.append(y)
        return matched_inten
    
    def Plot_Real(self, plotax, ions, spec):
        raw_scan = '.'.join(spec.split('.')[:-4])
        peaks = self.__read_one_spec__(raw_scan)
        xmz = np.array(peaks)[:,0]
        yint = np.array(peaks)[:,1]
        yint = yint[xmz <= self.max_plot_mz]
        xmz = xmz[xmz <= self.max_plot_mz]
        max_inten = np.max(yint)
        yint /= max_inten
        peaks = list(zip(xmz, yint))
        self.peak_hashing(peaks)
        
        max_charge = self.config.max_ion_charge if self.spec_charge >= self.config.max_ion_charge else self.spec_charge
        
        if self.show_plot: plotax.vlines(xmz, [0]*len(yint), yint, color = 'lightgray')
        
        valign = 'bottom'
        vmargin = 0.05
        
        for charge in range(1, self.spec_charge + 1):
            peak_idx, peak_inten = self.match_use_hashing([self.pepmass], charge, peaks)
            idx = peak_idx[0]
            if idx != -1:
                if self.show_plot: 
                    plotax.text( peaks[idx][0], peaks[idx][1] + vmargin, '{}({}+)'.format('M', charge),  color='gray', horizontalalignment="center",verticalalignment=valign)
        
        matched_inten = []
        for charge in range(1, max_charge + 1):
            # vmargin *= charge
            for ion_type in self.ion_types:
                target_ions = ions[ion_type]
                peak_idx, peak_inten = self.match_use_hashing(target_ions, charge, peaks)
                matched_inten.extend(peak_inten)
                for i in range(len(peak_idx)):
                    idx = peak_idx[i]
                    if idx != -1:
                        if peaks[idx][1] >= self.min_plot_inten:
                            if self.config.ion_terms[ion_type] == 'c': ion_idx = len(target_ions)-i
                            else: ion_idx = i+1
                            if self.show_plot: 
                                plotax.plot( [peaks[idx][0], peaks[idx][0]], [0, peaks[idx][1]], color=self.ion_color[ion_type], lw=2)
                                plotax.text( peaks[idx][0], peaks[idx][1] + vmargin, ion_type.format(str(ion_idx)+'+'*charge), rotation = 90, color=self.ion_color[ion_type], horizontalalignment="center",verticalalignment=valign)
        if self.show_plot: plotax.text(20,1.2,'x {:.2e}'.format(max_inten))
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
    
    # def match_real(self, ions, spec):
        # raw_scan = '.'.join(spec.split('.')[:-4])
        # peaks = self.__read_one_spec__(raw_scan)
        # xmz = np.array(peaks)[:,0]
        # yint = np.array(peaks)[:,1]
        # max_inten = np.max(yint)
        # yint /= max_inten
        # peaks = list(zip(xmz, yint))
        # self.peak_hashing(peaks)
        
        # max_charge = 2 if self.spec_charge > 2 else 1

        # matched_inten = []
        # for charge in range(1, max_charge + 1):
            # for ion_type in self.ion_types:
                # target_ions = ions[ion_type]
                # peak_idx, peak_inten = self.match_use_hashing(target_ions, charge, peaks)
                # matched_inten.extend(peak_inten)
        # return matched_inten
    
    # def match_predict(self, ions, predictions):
        # max_charge = 2 if self.spec_charge > 2 else 1
        # matched_inten = []
        # for charge in range(1, max_charge+1):
            # for ntype in range(len(self.ion_types)):
                # ion_type = self.ion_types[ntype]
                # target_ions = ions[ion_type]
                # for i in range(len(target_ions)):
                    # y = predictions[i,ntype*self.config.max_ion_charge+charge-1]
                    # matched_inten.append(y)
        # return matched_inten
    
    # def match_noplot(self, peptide, modinfo, spec):
        # raw_scan = '.'.join(spec.split('.')[:-4])
        # if not raw_scan in self.pf2idx:
            # print('no spec {} in pf2 file'.format(spec)) 
            # return
        
        # ions = {}
        # bions, pepmass = calc_b_ions(peptide, modinfo)
        # if 'b{}' in self.ion_types: ions['b{}'] = bions
        # if 'y{}' in self.ion_types: ions['y{}'] = calc_y_from_b(bions, pepmass)
        # if 'c{]' in self.ion_types: ions['c{}'] = calc_c_from_b(bions)
        # if 'z{}' in self.ion_types: ions['z{}'] = calc_z_from_b(bions, pepmass)
        
        # self.spec_charge = int(spec.split('.')[-3])
        
        # matched_inten1 = self.match_real(ions, spec)
        
        # x, mod_x = self.ion2vec.FeaturizeOnePeptide(peptide, modinfo, self.spec_charge)
        # if x is None: return None
        # if self.mod_mode:
            # predictions = self.model.predict([np.array([x]),np.array([mod_x])])[0,:,:]
        # else:
            # predictions = self.model.predict(np.array([x]))[0,:,:]
        # matched_inten2 = self.match_predict(ions, predictions)
        
        # if len(matched_inten1) < len(matched_inten2):
            # matched_inten2 = matched_inten2[:len(matched_inten1)]
            # print('[Warning] ion number is not equal')
        # elif len(matched_inten1) > len(matched_inten2):
            # matched_inten1 = matched_inten1[:len(matched_inten2)]
            # print('[Warning] ion number is not equal')
        # return pearsonr(np.array(matched_inten1), np.array(matched_inten2))[0]
    
    def plot(self, peptide, modinfo, spec, charge, nce, instrument):
        self.raw_scan = '.'.join(spec.split('.')[:-4])
        if not self.raw_scan in self.pf2idx:
            print('no spec {} in pf2 file'.format(spec)) 
            return
        print('{}-{} <-> {}'.format(peptide, modinfo, spec))
        
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
        
        matched_inten1 = self.Plot_Real(ax, ions, spec)
        self.max_real_inten = max(matched_inten1)
        
        buckets = self.ion2vec.FeaturizeOnePeptide_buckets(peptide, modinfo, self.spec_charge, nce, instrument)
        if buckets is None: return
        predictions = self.model.Predict(buckets)[len(peptide)][0][0,:,:]
        matched_inten2 = self.Plot_Predict(ax, ions, predictions)
        
        if len(matched_inten1) < len(matched_inten2):
            matched_inten2 = matched_inten2[:len(matched_inten1)]
            print('[Warning] ion number is not equal')
        elif len(matched_inten1) > len(matched_inten2):
            matched_inten1 = matched_inten1[:len(matched_inten2)]
            print('[Warning] ion number is not equal')
        PCC = pearsonr(np.array(matched_inten1), -np.array(matched_inten2))
        
        if self.show_plot:
            ax.text(200, 1.3, '{} ({}+), {}, R = {:.2f}'.format(peptide,self.spec_charge,modinfo,PCC[0]), fontsize = 14)
            
            ax.set_xlim(xmin = 0, xmax = max_plot_mz)
            ax.set_ylim(ymin = -1.2, ymax=1.4)
            ax.hlines([0], [0], [max_plot_mz])
            ax.set_xlabel('m/z')
            ax.set_ylabel('Relative Abundance')
            ylabels = ax.get_yticks().tolist()
            # ylabels = ["{:.0f}%".format(abs(float(label)*100)) if float(label) >= 0 else '' for label in ylabels]
            ylabels = ["{:.0f}%".format(abs(float(label)*100)) for label in ylabels]
            #ylabels = ['' for label in ylabels]
            ax.set_yticklabels(ylabels)
            
            print('R = {:.2f}'.format(PCC[0]))
        return (fig, PCC[0])
        
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
            
            fig, pcc = self.plot(peptide, mod, pep[2], pep[3], pep[4], pep[5])
            plt.tight_layout()
            mng = plt.get_current_fig_manager()
            #mng.window.showMaximized()
            #mng.resize(*mng.window.maxsize())
               
            
            plt.savefig(os.path.join(save_dir, "%s-%s-%s-R=%.3f.png"%(pep[2], peptide, short_mod, pcc)),format="png", dpi=220)
            plt.close()
            
    def batch_show(self, pf2, pep_list, save_dir = None):
        self.Readpf2(pf2)
        for pep in pep_list:
            self.plot(*pep)
            self.show()
    
    def show(self):
        if self.show_plot: plt.show()
        
    def pcc_batch(self, psmlist):
        show_plot_bak = self.show_plot
        self.show_plot = False
        pccs = []
        for psm in psmlist:
            fig, pcc = self.plot(peptide = psm[1], modinfo = '', spec = psm[0], charge = psm[3])
            if pcc is not None: pccs.append(pcc)
        self.show_plot = show_plot_bak
        return pccs

#if __name__ == "__main__":
    # jupyter usage
    # from model.tools.back2back_plot import bbplot
    # import model.fragmentation_config as fconfig
    # import model.lstm_unmod as lstm

    # lstm_unmod = lstm.IonLSTM(fconfig.HCD_Config())
    # lstm_unmod.LoadModel(r'.\h5\len20\2-layer-BiLSTM-QE-M-M.h5')

    # bb = bbplot(conf = fconfig.HCD_Config(), model = lstm_unmod)
    # bb.Readpf2(r'h:\zhouxiexuan\data2RawFiles\HCD\RAW\01625b_GA1-TUM_first_pool_1_01_01-3xHCD-1h-R1_HCDFT.pf2')
