
'''
Created on 2013-8-16

@author: RunData
'''

class AAMass(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.mass_H = 1.0078250321
        self.mass_O = 15.9949146221
        self.mass_proton = 1.007276
        self.mass_N = 14.0030740052
        self.mass_C = 12.00
        self.mass_isotope = 1.003
        
        self.mass_H2O = self.mass_H * 2 + self.mass_O
        self.mass_CO = self.mass_C + self.mass_O
        self.mass_CO2 = self.mass_C + self.mass_O * 2
        self.mass_NH = self.mass_N + self.mass_H
        self.mass_NH3 = self.mass_N + self.mass_H * 3
        self.mass_HO = self.mass_H + self.mass_O
        
        
        self.aa_mass_dict = {}
        self.aa_mass_dict['A'] = 71.037114
        self.aa_mass_dict['B'] = 0.
        self.aa_mass_dict['C'] = 103.009185
        self.aa_mass_dict['D'] = 115.026943
        self.aa_mass_dict['E'] = 129.042593
        self.aa_mass_dict['F'] = 147.068414
        self.aa_mass_dict['G'] = 57.021464
        self.aa_mass_dict['H'] = 137.058912
        self.aa_mass_dict['I'] = 113.084064
        self.aa_mass_dict['J'] = 114.042927
        self.aa_mass_dict['K'] = 128.094963
        self.aa_mass_dict['L'] = 113.084064
        self.aa_mass_dict['M'] = 131.040485
        self.aa_mass_dict['N'] = 114.042927
        self.aa_mass_dict['P'] = 97.052764
        self.aa_mass_dict['Q'] = 128.058578
        self.aa_mass_dict['R'] = 156.101111
        self.aa_mass_dict['S'] = 87.032028
        self.aa_mass_dict['T'] = 101.047679
        self.aa_mass_dict['U'] = 150.95363
        self.aa_mass_dict['V'] = 99.068414
        self.aa_mass_dict['X'] = 0.
        self.aa_mass_dict['W'] = 186.079313
        self.aa_mass_dict['Y'] = 163.06332
        self.aa_mass_dict['Z'] = 0.
        
        self.glyco_mass_dict = {}
        self.glyco_mass_dict["Xyl"] = 132.0422587452
        self.glyco_mass_dict["Hex"] = 162.0528234315
        self.glyco_mass_dict["dHex"] = 146.0579088094
        self.glyco_mass_dict["HexNAc"] = 203.07937253300003
        self.glyco_mass_dict["NeuAc"] = 291.09541652769997
        self.glyco_mass_dict["NeuGc"] = 307.09033114979997

        self.mod_mass_dict = {}
        # self.mod_mass_dict["Carbamidomethyl[C]"] = 57.021464
        # self.mod_mass_dict['Oxidation[M]'] = 15.994915
        self.__read_mod__()
    
    def __read_mod__(self):
        from ..modification import get_modification
        mod_dict = get_modification()
        for modname, modinfo in mod_dict.items():
            modinfo = modinfo.split(' ')
            modmass = float(modinfo[2])
            mod_neutral_loss = 0
            if modinfo[4] != '0':
                mod_neutral_loss = float(modinfo[5])
            self.mod_mass_dict[modname] = (modmass, mod_neutral_loss)
            
    def fix_C57(self):
        self.aa_mass_dict['C'] += 57.021464
        
    def fix_K8R10(self):
        self.aa_mass_dict['K'] += self.mod_mass_dict['Label_13C(6)15N(2)[K]'][0]
        self.aa_mass_dict['R'] += self.mod_mass_dict['Label_13C(6)15N(4)[R]'][0]
        
aamass = AAMass()