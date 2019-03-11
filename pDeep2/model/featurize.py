import math
import numpy as np
import warnings
import sys

# warnings.filterwarnings('error')

base_dtype = np.int8

def to_ndarray(bucket_value):
    ret = []
    for i in range(len(bucket_value[0])):
        if isinstance(bucket_value[0][i], np.ndarray):
            x = np.zeros((len(bucket_value),) + bucket_value[0][i].shape, dtype=bucket_value[0][i].dtype)
            for j in range(len(bucket_value)):
                x[j] = bucket_value[j][i]
            ret.append(x)
        else:
            dtype = base_dtype
            if isinstance(bucket_value[0][i], str): dtype = None
            elif isinstance(bucket_value[0][i], float): dtype = np.float32
            x = []
            for j in range(len(bucket_value)):
                x.append(bucket_value[j][i])
            ret.append(np.array(x, dtype = dtype))
    return ret
    
def to_numpy(buckets):
    ret_buckets = {}
    for peplen, bucket_value in buckets.items():
        ret_buckets[peplen] = to_ndarray(bucket_value)
    return ret_buckets

class Ion2Vector:
    def __init__(self, conf, prev = 1, next = 1):
        self.config = conf
        self.prev = prev
        self.next = next
        self.AAs = "ACDEFGHIKLMNPQRSTVWY"
        self.aa2vector = self.AAVectorDict()
        self.AA_idx = dict(zip(self.AAs, range(0,len(self.aa2vector))))
        self.__parse_mod__()
        self.fix_aa_mod = {}
        for item in self.config.fixmod:
            aa = item[item.find('[')+1:item.find(']')]
            self.fix_aa_mod[aa] = item
        
        self.max_samples = 100000000
        self.__parse_instrument__()
        
    def __parse_instrument__(self):
        self.instrument_feature = {}
        for i in range(self.config.max_instrument_num):
            feature = [0] * self.config.max_instrument_num
            feature[i] = 1
            if i < len(self.config.instrument_list):
                self.instrument_feature[self.config.instrument_list[i].lower()] = np.array(feature, dtype = base_dtype)
            if i == self.config.max_instrument_num - 1:
                self.instrument_feature['unknown'] = np.array(feature, dtype = base_dtype)
        
    def __parse_mod__(self):
        # feature_vector of mod: [#C, #H, #N, #O, #S, #P, #metal, #other]
        common = self.config.mod_common_elem
        metal = self.config.mod_metal_elem
        self.mod_feature_size = self.config.GetModFeatureSize()
        
        def parse_element(elem_str):
            feature = [0] * self.mod_feature_size
            elems = elem_str.split(')')[:-1]
            for elem in elems:
                chem, num = elem.split('(')
                num = int(num)
                
                try:
                    idx = common.index(chem)
                    feature[idx] = num
                except:
                    if chem in metal:
                        feature[-2] += num
                    else:
                        feature[-1] += num
            return np.array(feature, dtype=base_dtype)
        
        self.mod_feature = {}
        for modname, elem_str in self.config.mod_element.items():
            self.mod_feature[modname] = parse_element(elem_str.split(' ')[-1])

    def AAVectorDict(self):
        aa2vector_map = {}
        s = self.AAs
        v = [0]*len(s)
        v[0] = 1
        for i in range(len(s)):
            aa2vector_map[s[i]] = list(v)
            v[i],v[(i+1) % 20] = 0,1
        return aa2vector_map
        
    def CountVarMod(self, peptide, modlist):
        var_mod_count = 0
        for idx, modname in modlist:
            if modname in self.config.fixmod: continue
            elif modname in self.config.varmod: var_mod_count += 1
            else: return -1 #unexpected
        return var_mod_count
        
    def CheckLegalPeptide(self, peptide):
        for aa in peptide:
            if not aa in self.aa2vector: 
                print("[W] invalid aa '%s' in peptide '%s', ignored this peptide!"%(aa, peptide))
                return False
        return True

    def Parse_Ion2vector(self, peptide, ionidx, charge):
        seqidx = ionidx
        # look at the ion's previous "prev" N_term AA
        v = []
        for i in range(seqidx - self.prev, seqidx):
            if i < 0:
                v.extend([0]*len(self.aa2vector))
            else:
                v.extend(self.aa2vector[peptide[i]])
        # look at the ion's next "next" C_term AAs
        for i in range(seqidx, seqidx + self.next):
            if i >= len(peptide):
                v.extend([0]*len(self.aa2vector))
            else:
                v.extend(self.aa2vector[peptide[i]])
        
        #the number of each AA before "prev" in NTerm
        NTerm_AA_Count = [0]*len(self.aa2vector)
        for i in range(seqidx - self.prev):
            NTerm_AA_Count[self.AA_idx[peptide[i]]] += 1
        v.extend(NTerm_AA_Count)
        
        #the number of each AA after "next" in CTerm
        CTerm_AA_Count = [0]*len(self.aa2vector)
        for i in range(seqidx + self.next, len(peptide)):
            CTerm_AA_Count[self.AA_idx[peptide[i]]] += 1
        v.extend(CTerm_AA_Count)
        
        if ionidx == 1: NTerm = 1
        else: NTerm = 0
        if ionidx == len(peptide)-1: CTerm = 1
        else: CTerm = 0
        v.extend([NTerm,CTerm])
        
        # vchg = [0]*6
        # vchg[charge-1] = 1
        # v.extend(vchg)
        return np.array(v, dtype=base_dtype)
        
    def PaddingZero(self, ionvec, intenvec, modvec):
        for i in range(self.config.time_step - len(ionvec)):
            ionvec.append(np.array([0] * ( (len(ionvec[0]) ) ), dtype=base_dtype))
            intenvec.append(np.array([0] * (self.config.max_ion_charge * len(self.config.ion_types)), dtype=np.float32))
            modvec.append(np.array([0] * ( (len(modvec[0]) ) ), dtype=base_dtype))
        return (np.array(ionvec), np.array(intenvec), np.array(modvec))

    def FeaturizeOnePeptide(self, peptide, modinfo, charge, padding_zero = False):
        if len(peptide) > self.config.time_step + 1: return None
        
        mod_idx_feature = [np.array([0]*self.mod_feature_size, dtype=base_dtype) for i in range(len(peptide))]
        moditems = modinfo.split(';')
        unexpected_mod = False
        modlist = []
        var_mod_count = 0
        
        def feature_idx(idx, peplen):
            if idx ==0 or idx == 1: idx == 0
            elif idx >= peplen: idx = peplen-1
            else: idx -= 1
            return idx
            
        for mod in moditems:
            if not mod: continue
            modtmp = mod.split(',')
            idx = int(modtmp[0])
            modname = modtmp[1]
            modlist.append( (idx, modname) )
            if modname in self.config.fixmod:
                idx = feature_idx(idx, len(peptide))
                mod_idx_feature[idx] = self.mod_feature[modname]
            elif modname in self.config.varmod:
                idx = feature_idx(idx, len(peptide))
                mod_idx_feature[idx] = self.mod_feature[modname]
                var_mod_count += 1
            else:
                unexpected_mod = True
                break
        if var_mod_count < self.config.min_var_mod_num or var_mod_count > self.config.max_var_mod_num: return None
        if unexpected_mod: return None
        if not self.config.CheckFixMod(peptide, modlist): return None
        if not self.CheckLegalPeptide(peptide): return None
                
        x = []
        mod_x = []
        for site in range(1, len(peptide)):
            mod_x.append(np.append(mod_idx_feature[site-1], mod_idx_feature[site]))
            x.append(self.Parse_Ion2vector(peptide, site, charge))
        if padding_zero: x, __, mod_x = self.PaddingZero(x, [], mod_x)
        else: x, mod_x = np.array(x), np.array(mod_x)
        return x, mod_x
        
    def FeaturizeOnePeptide_buckets(self, peptide, modinfo, pre_charge, nce, instrument):
        peptide_info = "{}|{}|{}".format(peptide, modinfo, pre_charge)
        x = self.FeaturizeOnePeptide(peptide, modinfo, pre_charge)
        if x is None:
            print("[Error] Illegal peptide: {}".format(peptide_info))
            return None
        if instrument in self.instrument_feature:
            inst_feature = self.instrument_feature[instrument.lower()]
        else:
            inst_feature = self.instrument_feature['unknown']
        buckets = {len(peptide) : [(x[0], x[1], pre_charge, float(nce), inst_feature, peptide_info)]}
        return to_numpy(buckets)
    
    def Featurize_buckets_predict(self, peptide_list, nce, instrument):
        if instrument in self.instrument_feature:
            inst_feature = self.instrument_feature[instrument.lower()]
        else:
            inst_feature = self.instrument_feature['unknown']
        
        buckets = {}
        
        for peptide, modinfo, pre_charge in peptide_list:
            pre_charge = int(pre_charge)
            
            # print(items[0])
            x = self.FeaturizeOnePeptide(peptide, modinfo, pre_charge)
            if x is None: continue
            
            peplen = len(peptide)
            
            def ap(ap_to, add):
                ret = []
                for i in range(len(add)):
                    ret.append(np.append(ap_to[i], [add[i]], axis=0))
                return ret
            
            peptide_info = "{}|{}|{}".format(peptide, modinfo, pre_charge)
            if peplen in buckets:
                buckets[peplen].append((x[0], x[1], pre_charge, float(nce), inst_feature, peptide_info))
                #buckets[peplen] = ap(buckets[peplen], (x[0], x[1], pre_charge, peptide_info))
            else:
                buckets[peplen] = [(x[0], x[1], pre_charge, float(nce), inst_feature, peptide_info)]
                #buckets[peplen] = (np.array([x[0]]), np.array([x[1]]), np.array([pre_charge], dtype=base_dtype), np.array([peptide_info]))
        
        return to_numpy(buckets)
    
    def Featurize_buckets(self, ion_file, nce, instrument):
        f = open(ion_file)
        if instrument in self.instrument_feature:
            inst_feature = self.instrument_feature[instrument.lower()]
        else:
            inst_feature = self.instrument_feature['unknown']
        
        buckets = {}
        
        items = f.readline().strip().split('\t')
        headeridx = dict(zip(items, range(len(items))))
        
        charge_in_spec = True
        if "charge" in headeridx: charge_in_spec = False
        
        sample_count = 0
        while True:
            line = f.readline()
            if line == "": break
            type2inten = {}
            allinten = []
            items = line.split("\t")
            peptide = items[1]
            if charge_in_spec: pre_charge = int(items[0].split(".")[-3])
            else: pre_charge = int(items[headeridx["charge"]])
            modinfo = items[2]
            
            x = self.FeaturizeOnePeptide(peptide, modinfo, pre_charge)
            if x is None: continue
            
            type2inten = {}
            for ion_type in self.config.GetIonTypeNames():
                peaks = items[headeridx[ion_type]]
                if len(peaks) < 2: continue
                peaks = [peak.split(",") for peak in peaks.strip().strip(";").split(";")]
                type2inten.update(dict([(peak[0], float(peak[1])) for peak in peaks]))
            if len(type2inten) < len(peptide): continue
            
            intenvec = []            
            for site in range(1, len(peptide)):
                v = []
                for ion_type in self.config.ion_types:
                    ion_name = self.config.GetIonNameBySite(peptide, site, ion_type)
                    for charge in range(1, self.config.max_ion_charge+1):
                        ion_name_charge = ion_name + "+{}".format(charge)
                        if ion_name_charge in type2inten: v.append(type2inten[ion_name_charge])
                        else: v.append(0)
                intenvec.append(np.array(v, dtype=np.float32))
            
            intenvec = np.array(intenvec)
            intenvec /= np.max(intenvec)
            peplen = len(peptide)
                
            if peplen in buckets:
                buckets[peplen].append((x[0], x[1], pre_charge, float(nce), inst_feature, intenvec))
            else:
                buckets[peplen] = [(x[0], x[1], pre_charge, float(nce), inst_feature, intenvec)]
            sample_count += 1
            if sample_count >= self.max_samples: break
        f.close()
        
        return to_numpy(buckets)

