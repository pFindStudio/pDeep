import model.fragmentation_config as fconfig
import model.lstm_tf as lstm
import model.similarity_calc as sim_calc
from model.load_data import load_folder_as_buckets as load_folder
import model.load_data as load_data
from model.bucket_utils import merge_buckets, print_buckets, count_buckets
import model.evaluate as evaluate
import numpy as np
import tensorflow as tf
import time
import os

ion_types = ['b{}', 'y{}','b{}-ModLoss','y{}-ModLoss']
mod_syn_config = fconfig.HCD_CommonMod_Config()
mod_syn_config.varmod = [
    "Acetyl[K]",
    "Biotin[K]",
    "Butyryl[K]",
    "Crotonyl[K]",
    "Dimethyl[R]",
    "Dimethyl[R]",
    "Formyl[K]",
    "Glutaryl[K]",
    "Dicarbamidomethyl[K]",
    "Hydroxyisobutyryl[K]",
    "Malonyl[K]",
    "Methyl[K]",
    "Propionyl[K]",
    "Succinyl[K]",
    "Trimethyl[K]",
    "Oxidation[P]",
    "Deamidated[R]",
    "Dimethyl[K]",
    "Methyl[R]",
    "Nitro[Y]",
    "Phospho[Y]",
    "Phospho[S]",
    "Phospho[T]",
]
mod_syn_config.SetIonTypes(ion_types)
mod_syn_config.time_step = 100
mod_syn_config.min_var_mod_num = 1
mod_syn_config.max_var_mod_num = 1

pdeep = lstm.IonLSTM(mod_syn_config)

pdeep.learning_rate = 0.001
pdeep.layer_size = 256
pdeep.batch_size = 1024

pdeep.epochs = 20
n = 100000000

out_folder = './tf-models/model-180921-modloss'
model_folder = out_folder
model_name = 'pretrain-180921-modloss' #.ckpt

model_type = "AllMod"

pdeep.BuildTransferModel(os.path.join(out_folder, model_name+".ckpt"))

model_name += "-transfer.ckpt" # the model is saved as ckpt file

example,nce,instrument,conf = "./data/PT-21Mod",0.3,"Lumos",mod_syn_config
start_time = time.perf_counter()

buckets = {}
buckets = merge_buckets(buckets, load_folder(example, conf, nce = nce, instrument = instrument, max_n_samples = n))
print('[I] train data:')
print_buckets(buckets, print_peplen = False)
print(count_buckets(buckets))

pdeep.TrainModel(buckets, save_as = os.path.join(out_folder, model_name))

pdeep.close_session()
