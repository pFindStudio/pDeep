import numpy as np
import sys

_feature_name_list = ['x', 'mod_x', 'charge', 'nce', 'instrument', 'y']

def set_mod_zero_buckets(buckets):
    mod_idx = _feature_name_list.index("mod_x")
    for key, value in buckets.items():
        buckets[key][mod_idx] = np.zeros(value[mod_idx].shape)
    return buckets

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
        