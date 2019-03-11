import model.fragmentation_config as fconfig
import model.load_data as load_data
from model.bucket_utils import write_buckets, write_buckets_mgf
import numpy as np
import tensorflow as tf
import time

import sys

import model.lstm_tf as lstm
model_folder = './tf-models/model-180921-modloss/'
model = 'pretrain-180921-modloss.ckpt'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--nce", default=0.30, help="NCE", type=float)
parser.add_argument("-i", "--instrument", default="Lumos", help="instrument")
parser.add_argument("-in", "--input", default="peptide.txt", help="input peptide file")
parser.add_argument("-out", "--output", default="predict.txt", help="output mgf/txt file")
args = parser.parse_args()

print(vars(args))

CE = args.nce
instrument = args.instrument
in_file = args.input
out_file = args.output

mod_config = fconfig.HCD_CommonMod_Config()
mod_config.varmod.extend(['Phospho[S]','Phospho[T]','Phospho[Y]'])
mod_config.max_var_mod_num = 3

pdeep = lstm.IonLSTM(mod_config)

start_time = time.perf_counter()

buckets = load_data.load_peptide_file_as_buckets(in_file, mod_config, nce = CE, instrument = instrument)
read_time = time.perf_counter()

pdeep.LoadModel(model_file = model_folder + model)
output_buckets = pdeep.Predict(buckets)
predict_time = time.perf_counter()

write_buckets_mgf(out_file, buckets, output_buckets, mod_config)

print('read time = {:.3f}, predict time = {:.3f}'.format(read_time - start_time, predict_time - read_time))

pdeep.close_session()
