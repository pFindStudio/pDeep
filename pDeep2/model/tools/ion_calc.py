from .AAMass import aamass
from .modloss_priority import modloss_priority as prior

def calc_mod_mass_list(peptide, modinfo):
    items = modinfo.split(";")
    modlist = []
    for mod in items:
        if mod != '':
            site, modname = mod.split(",")
            site = int(site)
            modlist.append( (site, modname) )
    modlist.sort()
    
    modmass = [0]*(len(peptide)+2)
    lossmass = [0]*(len(peptide)+2)
    modname = [""]*(len(peptide)+2)
    for mod in modlist:
        modmass[mod[0]] = aamass.mod_mass_dict[mod[1]][0]
        lossmass[mod[0]] = aamass.mod_mass_dict[mod[1]][1]
        if mod[1] in prior: modname[mod[0]] = mod[1]
        else: modname[mod[0]] = "" # 0 priority
    return modmass,lossmass,modname
    
def calc_total_modmass(modinfo):
    items = modinfo.split(";")
    modlist = []
    for mod in items:
        if mod != '':
            site, modname = mod.split(",")
            site = int(site)
            modlist.append( (site, modname) )
    modmass = 0
    for mod in modlist:
        modmass += aamass.mod_mass_dict[mod[1]][0]
    return modmass
    
def calc_b_ions(peptide, modinfo):
    modmass_list, modloss_list, _ = calc_mod_mass_list(peptide, modinfo)
    b_ions = []
    mass_nterm = modmass_list[0]
    for i in range(len(peptide)-1):
        mass_nterm += aamass.aa_mass_dict[peptide[i]] + modmass_list[i+1]
        b_ions.append(mass_nterm)
    pepmass = b_ions[-1] + aamass.aa_mass_dict[peptide[-1]] + modmass_list[len(peptide)] + modmass_list[len(peptide)+1] + aamass.mass_H2O
    return b_ions, pepmass
    
def calc_pepmass(peptide, modinfo):
    modmass = calc_total_modmass(modinfo)
    mass_nterm = modmass
    for i in range(len(peptide)-1):
        mass_nterm += aamass.aa_mass_dict[peptide[i]]
    pepmass = mass_nterm + aamass.aa_mass_dict[peptide[-1]] + aamass.mass_H2O
    return pepmass
    
def calc_y_from_b(bions, pepmass):
    return [pepmass - b for b in bions]
    
def calc_c_from_b(bions, pepmass = 0):
    return [b + aamass.mass_NH3 for b in bions]
    
def calc_z_from_b(bions, pepmass):
    return [pepmass - b - aamass.mass_NH3 + aamass.mass_H for b in bions]
    
def calc_ion_modloss(ions, peptide, modinfo, N_term = True):
    # site_lossmass_list is list of multiple mod sites
    modmass_list, modloss_list, modname_list = calc_mod_mass_list(peptide, modinfo)
    ret = [0]*len(ions)
    if N_term:
        loss_nterm = modloss_list[0]
        modname_prev = modname_list[0]
        if modloss_list[1] != 0: 
            loss_nterm = modloss_list[1]
            modname_prev = modname_list[1]
        for i in range(len(ions)):
            if modloss_list[i+1] != 0: 
                if prior[modname_list[i+1]] > prior[modname_prev]:
                    loss_nterm = modloss_list[i+1]
                    modname_prev = modname_list[i+1]
            if loss_nterm != 0:
                ret[i] = ions[i] - loss_nterm
            else:
                ret[i] = 0
    else:
        loss_cterm = modloss_list[len(peptide)+1]
        modname_prev = modname_list[len(peptide)+1]
        if modloss_list[len(peptide)] != 0: 
            loss_cterm = modloss_list[len(peptide)]
            modname_prev = modname_list[len(peptide)]
        for i in range(len(ions)-1, -1, -1):
            if modloss_list[i+1] != 0: 
                if prior[modname_list[i+1]] > prior[modname_prev]:
                    loss_cterm = modloss_list[i+1]
                    modname_prev = modname_list[i+1]
            if loss_cterm != 0:
                ret[i] = ions[i] - loss_cterm
            else:
                ret[i] = 0
    return ret
    