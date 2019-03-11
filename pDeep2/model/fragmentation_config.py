from .modification import get_modification
import os

def get_all_mod_names(mod_dict):
    ret = []
    for key in mod_dict.keys():
        ret.append(key)
    return ret

class Common_Config(object):
    def __init__(self):
        self.ion_terms = {'b{}':'n','y{}':'c','c{}':'n','z{}':'c','b{}-ModLoss':'n','y{}-ModLoss':'c','b{}-H2O':'n','y{}-H2O':'c','b{}-NH3':'n','y{}-NH3':'c'}
        self.mod_element = get_modification()
        
        self.time_step = 100 #at most time_step+1 length peptides
        self.max_ion_charge = 2
        
        self.fragmentation = 'HCD'
        self.instrument_list = ['QE', 'Velos', 'Elite', 'Fusion', 'Lumos'] #they will be converted into lower case
        self.max_instrument_num = 8 # left 3 pos for the some other instruments
        self.enable_instrument_and_nce = True
        
        self.SetFixMod(['Carbamidomethyl[C]'])
        self.varmod = []
        self.min_var_mod_num = 0
        self.max_var_mod_num = 0
        
        # feature_vector of mod: [#C, #H, #N, #O, #S, #P, #metal, #other]
        self.mod_common_elem = ['C','H','N','O','S','P']
        self.mod_metal_elem = ['Na','Ca','Fe','K','Mg','Cu']
        
        self.SetIonTypes(['b{}','y{}'])
        
    def SaveConfig(self, model_path):
        # model_name.fconfig
        with open(model_path + ".pdeep_config",'w') as f:
            folder, model_name = os.path.split(model_path)
            f.write("#### configuration of model %s. ####\n#### Do not modify this config file ####\n"%model_name)
            f.write("ion_types=%s\n"%(','.join(self.ion_types)))
            f.write("fixmod=%s\n"%(','.join(self.fixmod)))
            f.write("varmod=%s\n"%(','.join(self.varmod)))
            f.write("min_var_mod_num=%d\n"%self.min_var_mod_num)
            f.write("max_var_mod_num=%d\n"%self.max_var_mod_num)
            f.write("max_ion_charge=%d\n"%self.max_ion_charge)
            f.write("time_step=%d\n"%self.time_step)
            f.write("fragmentation=%s\n"%self.fragmentation)
            f.write("instrument_list=%s\n"%(','.join(self.instrument_list)))
            f.write("max_instrument_num=%d\n"%self.max_instrument_num)
            f.write("enable_instrument={}\n"%self.enable_instrument)
            f.write("enable_nce={}\n".format(self.enable_nce))
            f.write("mod_common_elem=%s\n"%(','.join(self.mod_common_elem)))
            f.write("mod_metal_elem=%s\n"%(','.join(self.mod_metal_elem)))
            
    def LoadConfig(self, model_path):
        contents = {}
        with open(model_path + ".fconfig") as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('#'):
                    items = line.split('=')
                    if len(items) == 2:
                        contents[items[0].strip()] = items[1].strip()
            
            if "ion_types" in contents:
                self.ion_types = contents["ion_types"].split(',')
            if "fixmod" in contents:
                self.fixmod = contents["fixmod"].split(',')
            if "varmod" in contents:
                self.varmod = contents["varmod"].split(',')
            if "min_var_mod_num" in contents:
                self.min_var_mod_num = int(contents["min_var_mod_num"])
            if "max_var_mod_num" in contents:
                self.max_var_mod_num = int(contents["max_var_mod_num"])
            if "max_ion_charge" in contents:
                self.max_ion_charge = int(contents["max_ion_charge"])
            if "time_step" in contents:
                self.time_step = int(contents["time_step"])
            if "fragmentation" in contents:
                self.fragmentation = contents["fragmentation"]
            if "instrument_list" in contents:
                self.instrument_list = contents["instrument_list"].split(',')
            if "max_instrument_num" in contents:
                self.max_instrument_num = int(contents["max_instrument_num"])
            if "enable_instrument" in contents:
                self.enable_instrument = bool(contents["enable_instrument"])
            if "enable_nce" in contents:
                self.enable_nce = bool(contents["enable_nce"])
            if "mod_common_elem" in contents:
                self.mod_common_elem = contents["mod_common_elem"].split(',')
            if "mod_metal_elem" in contents:
                self.mod_metal_elem = contents["mod_metal_elem"].split(',')
    
    def GetModFeatureSize(self):
        return len(self.mod_common_elem) + 2
        
    def GetTFOutputSize(self):
        return len(self.ion_types)*self.max_ion_charge
    
    def SetMaxProductIonCharge(self, max_charge = 2):
        self.max_ion_charge = max_charge
        
    def SetTimeStep(self, time_step):
        self.time_step = time_step
    
    def SetVarMod(self, modlist, min_var_num, max_var_num):
        # ['Oxidation[M]', 'Deamidated[N]']
        self.varmod = modlist
        self.min_var_mod_num = min_var_num
        self.max_var_mod_num = max_var_num
        
    def SetFixMod(self, modlist):
        self.fixmod = modlist
        self.fix_aa_mod = {}
        for item in self.fixmod:
            aa = item[item.find('[')+1:item.find(']')]
            self.fix_aa_mod[aa] = item
            
    def CheckFixMod(self, peptide, modlist):
        return self.CheckFixMod_fixpass(peptide, modlist)
        
    def CheckFixMod_fixall(self, peptide, modlist):
        fixed = [0]*len(peptide)
        for i in range(len(peptide)):
            if peptide[i] in self.fix_aa_mod: fixed[i] = 1
        for idx, modname in modlist:
            if idx > 0 and idx <= len(peptide):
                if peptide[idx-1] in self.fix_aa_mod:
                    if modname == self.fix_aa_mod[peptide[idx-1]]: fixed[idx-1] = 0
        return sum(fixed) == 0
        
    def CheckFixMod_fixpass(self, peptide, modlist):
        return True
    
    def SetIonTypes(self, ion_types):
        self.ion_types = ion_types
            
        self.SetPredictIonIndex() # predicted intensity index of different ions in ndarray
        
    def GetIonTypeNames(self):
        return [ion_type.format("") for ion_type in self.ion_types]
        
    def GetIonNameBySite(self, peptide, site, ion_type):
        if self.ion_terms[ion_type] == 'c': return ion_type.format(len(peptide) - site)
        else: return ion_type.format(site)
        
    def GetIntenIdx(self, iontype):
        return self.pred_ion_idx[iontype]
    
    def SetPredictIonIndex(self):
        self.pred_ion_idx = dict(zip(self.ion_types, range(len(self.ion_types))))
    
    def GetIonIndexByIonType(self, ion_type, ion_charge):
        if ion_charge > self.max_ion_charge: return None
        if not ion_type in self.pred_ion_idx: return None
        return self.pred_ion_idx[ion_type]*self.max_ion_charge + ion_charge - 1
        
    def GetIntenFromNDArrayByLossName(self, inten_ndarray, loss_name = None):
        # loss_name = None: noloss, "ModLoss", "H2O", "NH3"
        # shape of inten_ndarray: (peptides, cleavage_sites, ion_types)
        idxes = []
        if loss_name == None: loss_name = "{}"
        for ion_type in self.ion_types:
            if ion_type.endswith(loss_name):
                for ion_charge in range(1, self.max_ion_charge+1):
                    idxes.append(self.GetIonIndexByIonType(ion_type, ion_charge))
        return inten_ndarray[:,:,idxes]

class HCD_test_yion_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['y{}'])
        
class HCD_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()

class HCD_ProteomeTools_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()

class ETD_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['c{}','z{}'])
        self.fragmentation = 'ETD'

class ETD_ProteomeTools_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['c{}','z{}'])
        self.fragmentation = 'ETD'
        
class EThcD_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['b{}','y{}','c{}','z{}'])
        self.fragmentation = 'EThcD'
        
class EThcD_ProteomeTools_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['b{}','y{}','c{}','z{}'])
        self.fragmentation = 'EThcD'

class HCD_pho_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.varmod.extend(['Phospho[S]','Phospho[T]','Phospho[Y]'])
        self.min_var_mod_num = 0
        self.max_var_mod_num = 2
        
class HCD_oxM_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['b{}','y{}'])
        self.varmod.extend(['Oxidation[M]'])
        self.min_var_mod_num = 0
        self.max_var_mod_num = 2
        
class HCD_CommonMod_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['b{}','y{}'])
        self.varmod = ['Oxidation[M]', 'Deamidated[N]', 'Deamidated[Q]', 'Acetyl[ProteinN-term]', 'Acetyl[AnyN-term]', 'Formyl[AnyN-term]', 'Gln->pyro-Glu[AnyN-termQ]']
        self.min_var_mod_num = 0
        self.max_var_mod_num = 2
        
class HCD_AllMod_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['b{}','y{}'])
        self.varmod = get_all_mod_names(self.mod_element)
        self.min_var_mod_num = 0
        self.max_var_mod_num = 2
        
class ETD_pho_Config(Common_Config):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.SetIonTypes(['c{}','z{}'])
        self.varmod = ['Phospho[S]','Phospho[T]','Phospho[Y]']
        self.min_var_mod_num = 1
        self.max_var_mod_num = 1
        self.fragmentation = 'ETD'
        
