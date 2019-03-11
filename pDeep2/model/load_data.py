import os

from .featurize import Ion2Vector
from .bucket_utils import merge_buckets

def load_plabel_as_buckets(filenames, config, nce, instrument, max_n_samples = 10000000000):
    ion2vec = Ion2Vector(conf = config, prev = 1, next = 1)
    ion2vec.max_samples = max_n_samples
    buckets = {}
    count = 0
    for filename in filenames:
        count += 1
        print("%dth plabel"%count, end = "\r")
        _buckets = ion2vec.Featurize_buckets(filename, nce, instrument)
        buckets = merge_buckets(buckets, _buckets)
    return buckets
    
def load_folder_as_buckets(dataset_folder, config, nce, instrument = 'QE', max_n_samples = 10000000000):
    print("Loading %s .."%dataset_folder)
    filenames = []
    for input_file in os.listdir(dataset_folder):
        if input_file.endswith(".plabel"):
            filenames.append(os.path.join(dataset_folder, input_file))
    return load_plabel_as_buckets(filenames, config, nce, instrument, max_n_samples)
    
def load_files_as_buckets(filenames, config, nce, instrument = 'QE', max_n_samples = 10000000000):
    print("Loading data from files...")
    return load_plabel_as_buckets(filenames, config, nce, instrument, max_n_samples)
    
# format 'peptide	modification	charge'
def load_peptide_file_as_buckets(filename, config, nce, instrument = 'QE'):
    peptide_list = []
    with open(filename) as f:
        head = f.readline()
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line) == 0: continue
        peptide_list.append(line.split("\t"))
    return load_peptides_as_buckets(peptide_list, config, nce, instrument)

# format (peptide,modification,charge)
def load_peptides_as_buckets(peptide_list, config, nce, instrument = 'QE'):
    ion2vec = Ion2Vector(conf = config, prev = 1, next = 1)
    buckets = ion2vec.Featurize_buckets_predict(peptide_list, nce, instrument)
    return buckets