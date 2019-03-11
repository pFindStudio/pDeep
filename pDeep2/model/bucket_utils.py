import numpy as np
import sys
from .tools.ion_calc import *

_feature_name_list = ['x', 'mod_x', 'charge', 'nce', 'instrument', 'y']

def set_mod_zero_buckets(buckets):
    mod_idx = _feature_name_list.index("mod_x")
    for key, value in buckets.items():
        buckets[key][mod_idx] = np.zeros(value[mod_idx].shape)
    return buckets
def write_buckets_mgf(outfile, buckets, predict_buckets, fconfig, iontypes = ['b{}', 'y{}']):
    def write_one(f, pepinfo, pred):
        peptide, mod, charge = pepinfo.split("|")
        f.write('BEGIN IONS\n')
        f.write('TITLE=' + pepinfo + '\n')
        f.write('CHARGE=' + charge + '\n')
        pre_charge = int(charge)
        f.write('pepinfo=' + pepinfo + '\n')
        
        ions = {}
        bions, pepmass = calc_b_ions(peptide, mod)
        
        if 'b{}' in fconfig.ion_types and 'b{}' in iontypes: ions['b{}'] = bions
        if 'y{}' in fconfig.ion_types and 'y{}' in iontypes: ions['y{}'] = calc_y_from_b(bions, pepmass)
        if 'c{}' in fconfig.ion_types and 'c{}' in iontypes: ions['c{}'] = calc_c_from_b(bions)
        if 'z{}' in fconfig.ion_types and 'z{}' in iontypes: ions['z{}'] = calc_z_from_b(bions, pepmass)
        if 'b{}-ModLoss' in fconfig.ion_types and 'b{}-ModLoss' in iontypes: ions['b{}-ModLoss'] = calc_ion_modloss(bions, peptide, modinfo, N_term = True)
        if 'y{}-ModLoss' in fconfig.ion_types and 'y{}-ModLoss' in iontypes: ions['y{}-ModLoss'] = calc_ion_modloss(ions['y{}'], peptide, modinfo, N_term = False)
        
        max_charge = fconfig.max_ion_charge if pre_charge >= fconfig.max_ion_charge else pre_charge
        
        peak_list = []
        
        for ion_type in ions.keys():
            x_ions = np.array(ions[ion_type])
            for charge in range(1, max_charge+1):
                intens = pred[:,fconfig.GetIonIndexByIonType(ion_type, charge)]
                f.write('{}={}\n'.format(ion_type.format("+"+str(charge)), ','.join(['%.5f'%inten for inten in intens])))
                x_types = [fconfig.GetIonNameBySite(peptide, site, ion_type)+"+"+str(charge) for site in range(1,len(peptide))]
                peak_list.extend(zip(x_ions / charge + aamass.mass_proton, intens, x_types))
        
        pepmass = pepmass / pre_charge + aamass.mass_proton
        f.write("PEPMASS=%.5f\n"%pepmass)
        
        peak_list.sort()
        for mz, inten, ion_type in peak_list:
            if inten > 1e-8: f.write("%f %.8f %s\n"%(mz, inten, ion_type))
            
        f.write('END IONS\n')
    
    with open(outfile, 'w') as f:
        for key, value in buckets.items():
            preds = predict_buckets[key][-1]
            for i in range(value[-1].shape[0]):
                write_one(f, value[-1][i], preds[i])
def write_buckets(outfile, buckets, predict_buckets, iontypes = ['b+1', 'b+2', 'y+1', 'y+2']):
    def write_one(f, pepinfo, pred):
        f.write('BEGIN IONS\n')
        f.write('pepinfo=' + pepinfo + '\n')
        for i in range(len(iontypes)):
            f.write('{}={}\n'.format(iontypes[i], ','.join(['%.5f'%inten for inten in pred[:,i]])))
        f.write('END IONS\n')
    
    with open(outfile, 'w') as f:
        for key, value in buckets.items():
            preds = predict_buckets[key][-1]
            for i in range(value[-1].shape[0]):
                write_one(f, value[-1][i], preds[i])
                
write_predict = write_buckets

def print_buckets(buckets, print_peplen = True, print_file = sys.stdout):
    total_size = 0
    for key, value in buckets.items():
        if print_peplen:
            str = '[I] '
            str += 'peplen = %d'%key
            for i in range(len(value)):
                str += ', x{}.shape = {}'.format(i, value[i].shape)
            print(str, file = print_file)
        total_size += value[0].shape[0]
    print('[I] total data size = {}'.format(total_size), file = print_file)
    
    
def count_buckets(buckets):
    ret = {}
    ret["total"] = 0
    for key, value in buckets.items():
        ret[str(key)] = value[0].shape[0]
        ret["total"] += value[0].shape[0]
        
    return ret
    
def fixed_length_buckets(buckets, time_step = 100, time_step_dim = 1):
    new_buckets = {}
    x_idx = _feature_name_list.index("x")
    mod_idx = _feature_name_list.index("mod_x")
    for key, features in buckets.items():
        if key - 1 == time_step:
            new_buckets[key] = features
        elif key - 1 < time_step:
            shape = features.shape
            shape[time_step_dim] = time_step
            padding_dim = [(0,time_step - features.shape[time_step_dim]) if i == time_step_dim else (0,0) for i in range(len(shape))] # only pad for time_step_dim
            features[x_idx] = np.pad(features[x_idx], padding_dim, mode = 'constant', constant_values = 0)
            features[mod_idx] = np.pad(features[mod_idx], padding_dim, mode = 'constant', constant_values = 0)
            new_buckets[key] = features
        else:
            pass #ignore key - 1 > time_step
    return new_buckets
    
def merge_buckets(buckets, buckets_new):
    def merge_buckets_tuples(t1, t2):
        ret = []
        for i in range(len(t1)):
            ret.append(np.append(t1[i], t2[i], axis = 0))
        return ret
    
    for key, value in buckets_new.items():
        if key in buckets:
            buckets[key] = merge_buckets_tuples(buckets[key], value)
        else:
            buckets[key] = value
    return buckets

class Bucket_Batch(object):
    def __init__(self, buckets, batch_size = 1024, shuffle = True):
        self._buckets = buckets
        self._bucket_keys = np.array(list(buckets.keys()), dtype=np.int32)
        self._shuffle = shuffle
        self._batch_size = batch_size
        self.reset_batch()
        
        if len(buckets) != 0:
            first_key = next(iter(buckets))
            self._tuple_len = len(buckets[first_key])
        else:
            self._tuple_len = 0
        self._feature_name_list = _feature_name_list
        self._tuple_idx = dict(zip(self._feature_name_list, range(len(self._feature_name_list))))
        self._tuple_idx['peplen'] = -1
    
    def get_data_from_batch(self, batch, name):
        return batch[self._tuple_idx[name]]
    
    def reset_batch(self):
        if self._shuffle:
            self._bucket_keys = np.random.permutation(self._bucket_keys)
        self._cur_key = 0
        self.reset_cur_bucket()
        
    def reset_cur_bucket(self):
        if self._cur_key < len(self._bucket_keys):
            self._cur_bucket_idxs = np.arange(len(self._buckets[self._bucket_keys[self._cur_key]][0]))
            if self._shuffle:
                self._cur_bucket_idxs = np.random.permutation(self._cur_bucket_idxs)
            self.reset_cur_idxs()
            
    def reset_cur_idxs(self):
        self._cbs = 0
        self._cbe = self._batch_size
        if self._cbe > len(self._cur_bucket_idxs):
            self._cbe = len(self._cur_bucket_idxs)
    
    def get_next_batch(self):
        if self._cur_key >= len(self._bucket_keys):
            return None
        peplen = self._bucket_keys[self._cur_key]
        ret = []
        def get_one_batch(bucket_value):
            ret = []
            for i in range(len(bucket_value)):
                ret.append(bucket_value[i][self._cur_bucket_idxs[self._cbs:self._cbe]])
            return ret
        ret = get_one_batch(self._buckets[peplen])
        ret.append(np.array([peplen] * (self._cbe - self._cbs), dtype=np.int32))
        if self._cbe == len(self._cur_bucket_idxs):
            self._cur_key += 1
            self.reset_cur_bucket()
        else:
            self._cbs = self._cbe
            self._cbe += self._batch_size
            if self._cbe > len(self._cur_bucket_idxs):
                self._cbe = len(self._cur_bucket_idxs)
        return ret
        