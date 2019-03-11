import model.fragmentation_config as fconfig
import model.lstm_tf as lstm
from model.load_data import load_folder_as_buckets as load_folder
from model.bucket_utils import merge_buckets, print_buckets, count_buckets
import numpy as np
import tensorflow as tf
import os

ion_types = ['b{}', 'y{}']
mod_config = fconfig.HCD_CommonMod_Config()
mod_config.SetFixMod(['Carbamidomethyl[C]'])
mod_config.varmod = ["Oxidation[M]"]
mod_config.SetIonTypes(ion_types)
mod_config.time_step = 100
mod_config.min_var_mod_num = 0
mod_config.max_var_mod_num = 2

pdeep = lstm.IonLSTM(mod_config)

pdeep.learning_rate = 0.001
pdeep.layer_size = 256
pdeep.batch_size = 1024
pdeep.BuildModel(input_size = 98, output_size = mod_config.GetTFOutputSize(), nlayers = 2)

pdeep.epochs = 100
n = 100000000

out_folder = './tf-models/example/'
model_name = 'example.ckpt' # the model is saved as ckpt file

try:
    os.makedirs(out_folder)
except:
    pass
    
    
buckets = {}
PT_NCE30 = "./data/ProteomeTools" # folder containing plabel files
buckets = merge_buckets(buckets, load_folder(PT_NCE30, mod_config, nce = 0.30, instrument = 'Lumos', max_n_samples = n))
# you can add more plabel-containing folders here

print('[I] train data:')
print_buckets(buckets, print_peplen = False)
buckets_count = count_buckets(buckets)
print(buckets_count)
print(buckets_count["total"])


pdeep.TrainModel(buckets, save_as = os.path.join(out_folder, model_name))

pdeep.close_session()
